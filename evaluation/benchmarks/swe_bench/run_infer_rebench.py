import asyncio
import copy
import json
import os
import tempfile
from typing import Any, Literal, Awaitable, Callable, TextIO

import pandas as pd
import toml
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader

import openhands.agenthub
from evaluation.benchmarks.swe_bench.binary_patch_utils import (
    remove_binary_diffs,
    remove_binary_files_from_git,
)
from evaluation.benchmarks.swe_bench.resource.mapping import (
    get_instance_resource_factor,
)
from evaluation.benchmarks.swe_bench.resource.swt_bench_constants import (
    MAP_REPO_TO_INSTALL,
    MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE,
    MAP_VERSION_TO_INSTALL,
)
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    check_maximum_retries_exceeded,
    codeact_user_response,
    get_default_sandbox_config_for_eval,
    get_metrics,
    get_openhands_config_for_eval,
    is_fatal_evaluation_error,
    make_metadata,
    reset_logger_for_multiprocessing,
    update_llm_config_for_completions_logging,
    _process_instance_wrapper_mp,
    _process_instance_wrapper,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
    get_agent_config_arg,
    get_evaluation_parser,
    get_llm_config_arg,
    get_llms_for_routing_config,
    get_model_routing_config_arg,
)
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.critic import AgentFinishedCritic
from openhands.events.action import CmdRunAction, FileReadAction, MessageAction
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
)
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.shutdown_listener import sleep_if_should_continue

from swebenchrebench.harness.run_evaluation import run_instance, make_test_spec
import docker
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
ENABLE_LLM_EDITOR = os.environ.get('ENABLE_LLM_EDITOR', 'false').lower() == 'true'
BenchMode = Literal['swe', 'swt', 'swt-ci']

# Global variable to track dataset type
DATASET_TYPE = 'SWE-bench'


def set_dataset_type(dataset_name: str) -> str:
    """Set dataset type based on dataset name."""
    global DATASET_TYPE
    name_lower = dataset_name.lower()

    if 'swe-gym' in name_lower:
        DATASET_TYPE = 'SWE-Gym'
    elif 'swe-bench-live' in name_lower:
        DATASET_TYPE = 'SWE-bench-Live'
    elif 'rebench' in name_lower:
        DATASET_TYPE = 'SWE-rebench'
    elif 'multimodal' in name_lower:
        DATASET_TYPE = 'Multimodal'
    else:
        DATASET_TYPE = 'SWE-bench'

    logger.info(f'Dataset type set to: {DATASET_TYPE}')


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': codeact_user_response,
}


def _get_swebench_workspace_dir_name(instance: pd.Series) -> str:
    if DATASET_TYPE == 'SWE-bench-Live':
        return instance.instance_id
    else:
        return f'{instance.repo}__{instance.version}'.replace('/', '__')


def get_instruction(instance: pd.Series, metadata: EvalMetadata) -> MessageAction:
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    mode = metadata.details['mode']
    llm_model = metadata.llm_config.model

    # Determine the template file based on mode and LLM
    if metadata.instruction_template_name:
        template_name = metadata.instruction_template_name
    elif mode.startswith('swt'):
        template_name = 'swt.j2'
    elif mode == 'swe':
        if 'gpt-4.1' in llm_model:
            template_name = 'swe_gpt4.j2'
        else:
            template_name = (
                'swe_default.j2'  # Default for 'swe' mode (regular swe-bench)
            )
    else:
        # Fallback or error handling if mode is unexpected
        logger.error(f'Unexpected evaluation mode: {mode}. Falling back to default.')
        template_name = 'swe_default.j2'

    logger.debug(f'Using instruction template file: {template_name}')
    # Set up Jinja2 environment
    # Assuming templates are in 'evaluation/benchmarks/swe_bench/prompts' relative to this script
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        'instance': instance,
        'workspace_dir_name': workspace_dir_name,
        'metadata': metadata,  # Pass metadata if needed in templates
    }

    # Add specific context for swt-ci mode if needed
    if mode == 'swt-ci':
        context['test_instructions'] = (
            f'The following command can be used to run the tests: `{list(MAP_REPO_TO_TEST_FRAMEWORK_VERBOSE[instance.repo].values())[0]}`. Make sure they fail in the expected way.\n'
        )
    else:
        context['test_instructions'] = ''  # Ensure it's defined for other modes

    # Render the instruction
    instruction = template.render(context)

    if RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
        )

    if 'image_assets' in instance:
        assets = json.loads(instance['image_assets'])
        assert 'problem_statement' in assets, (
            'problem_statement is required in image_assets'
        )
        image_urls = assets['problem_statement']
        return MessageAction(content=instruction, image_urls=image_urls)
    return MessageAction(content=instruction)


# TODO: migrate all swe-bench docker to ghcr.io/openhands
DEFAULT_DOCKER_IMAGE_PREFIX = os.environ.get(
    'EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/'
)
logger.info(f'Default docker image prefix: {DEFAULT_DOCKER_IMAGE_PREFIX}')


def get_instance_docker_image(
    instance_id: str,
    swebench_official_image: bool = False,
) -> str:
    global DEFAULT_DOCKER_IMAGE_PREFIX
    if swebench_official_image:
        # Official SWE-Bench image
        # swebench/sweb.eval.x86_64.django_1776_django-11333:v1
        # SWE-bench-Live uses the same naming convention as SWE-Bench
        if DATASET_TYPE == 'SWE-bench-Live':
            docker_image_prefix = 'docker.io/starryzhang/'
        elif DATASET_TYPE == 'SWE-bench':
            docker_image_prefix = 'docker.io/swebench/'
        elif DATASET_TYPE == 'SWE-rebench':
            docker_image_prefix = 'docker.io/swerebench/'
        repo, name = instance_id.split('__')
        image_name = f'{docker_image_prefix.rstrip("/")}/sweb.eval.x86_64.{repo}_1776_{name}:latest'.lower()
        logger.debug(f'Using official SWE-Bench image: {image_name}')
        DEFAULT_DOCKER_IMAGE_PREFIX = docker_image_prefix
        return image_name
    else:
        # OpenHands version of the image
        docker_image_prefix = DEFAULT_DOCKER_IMAGE_PREFIX
        image_name = 'sweb.eval.x86_64.' + instance_id
        image_name = image_name.replace(
            '__', '_s_'
        )  # to comply with docker image naming convention
        return (docker_image_prefix.rstrip('/') + '/' + image_name).lower()


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    # We use a different instance image for the each instance of swe-bench eval
    use_swebench_official_image = DATASET_TYPE != 'SWE-Gym'

    base_container_image = get_instance_docker_image(
        instance['instance_id'],
        swebench_official_image=use_swebench_official_image,
    )
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
    )

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    # Add platform to the sandbox config to solve issue 4401
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.remote_runtime_resource_factor = get_instance_resource_factor(
        dataset_name=metadata.dataset,
        instance_id=instance['instance_id'],
    )

    config = get_openhands_config_for_eval(
        metadata=metadata,
        enable_browser=RUN_WITH_BROWSING,
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox_config=sandbox_config,
    )

    config.file_store_path=os.environ.get("FILE_STORE_PATH", '/shared_workspace/yanruo/github/OpenHands/tmp')

    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    # get 'draft_editor' config if exists
    config.set_llm_config(get_llm_config_arg('draft_editor'), 'draft_editor')
    config.set_llm_config(True, 'disable_stop_word')

    model_routing_config = get_model_routing_config_arg()
    model_routing_config.llms_for_routing = (
        get_llms_for_routing_config()
    )  # Populate with LLMs for routing from config.toml file

    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=ENABLE_LLM_EDITOR,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
        model_routing=model_routing_config,
        enable_history_truncation=False,
    )
    config.set_agent_config(agent_config)

    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required
    metadata: EvalMetadata,
):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)
    obs: CmdOutputObservation

    # Set instance id and git configuration
    action = CmdRunAction(
        command=f"""echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && echo "alias git='git --no-pager'" >> ~/.bashrc && git config --global core.pager "" && git config --global diff.binary false"""
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to export SWE_INSTANCE_ID and configure git: {str(obs)}',
    )

    action = CmdRunAction(command="""export USER=$(whoami); echo USER=${USER} """)
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to export USER: {str(obs)}')

    # inject the init script
    script_dir = os.path.dirname(__file__)

    # inject the instance info
    action = CmdRunAction(command='mkdir -p /swe_util/eval_data/instances')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to create /swe_util/eval_data/instances: {str(obs)}',
    )

    swe_instance_json_name = 'swe-bench-instance.json'
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the full path for the desired file name within the temporary directory
        temp_file_path = os.path.join(temp_dir, swe_instance_json_name)
        # Write to the file with the desired name within the temporary directory
        with open(temp_file_path, 'w') as f:
            if not isinstance(instance, dict):
                json.dump([instance.to_dict()], f)
            else:
                json.dump([instance], f)

        # Copy the file to the desired location
        runtime.copy_to(temp_file_path, '/swe_util/eval_data/instances/')

        # inject the instance swe entry
        if DATASET_TYPE == 'SWE-bench-Live':
            entry_script_path = 'instance_swe_entry_live.sh'
        elif DATASET_TYPE == 'SWE-rebench':
            entry_script_path = 'instance_swe_entry_rebench.sh'
        else:
            entry_script_path = 'instance_swe_entry.sh'
        runtime.copy_to(
            str(os.path.join(script_dir, f'scripts/setup/{entry_script_path}')),
            '/swe_util/',
        )

    action = CmdRunAction(command='cat ~/.bashrc')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to cat ~/.bashrc: {str(obs)}')

    action = CmdRunAction(command='source ~/.bashrc')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if isinstance(obs, ErrorObservation):
        logger.error(f'Failed to source ~/.bashrc: {str(obs)}')
    assert_and_raise(obs.exit_code == 0, f'Failed to source ~/.bashrc: {str(obs)}')

    action = CmdRunAction(command=f'source /swe_util/{entry_script_path}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to source /swe_util/{entry_script_path}: {str(obs)}',
    )

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )

    action = CmdRunAction(command='git reset --hard')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to git reset --hard: {str(obs)}')

    action = CmdRunAction(
        command='for remote_name in $(git remote); do git remote remove "${remote_name}"; done'
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to remove git remotes: {str(obs)}')

    if metadata.details['mode'] == 'swt-ci':
        # set up repo
        setup_commands = []
        if instance['repo'] in MAP_REPO_TO_INSTALL:
            setup_commands.append(MAP_REPO_TO_INSTALL[instance['repo']])

        # Run pre-install set up if provided
        install = MAP_VERSION_TO_INSTALL.get(instance['repo'], {}).get(
            instance['version'], []
        )
        if 'pre_install' in install:
            for pre_install in install['pre_install']:
                setup_commands.append(pre_install)

        if 'install' in install:
            setup_commands.append(install['install'])

        for command in setup_commands:
            action = CmdRunAction(command=command)
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if DATASET_TYPE != 'Multimodal' and DATASET_TYPE != 'SWE-bench-Live':
        # Only for non-multimodal datasets, we need to activate the testbed environment for Python
        # SWE-Bench multimodal datasets and SWE-bench-Live are not using the testbed environment
        action = CmdRunAction(command='which python')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0 and 'testbed' in obs.content,
            f'Expected to find python interpreter from testbed, but got: {str(obs)}',
        )

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,  # this argument is not required, but it is used to get the workspace_dir_name
) -> dict[str, Any]:
    """Complete the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    If you need to do something in the sandbox to get the correctness metric after
    the agent has run, modify this function.
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation
    workspace_dir_name = _get_swebench_workspace_dir_name(instance)

    action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code == -1:
        # The previous command is still running
        # We need to kill previous command
        logger.info('The previous command is still running, trying to kill it...')
        action = CmdRunAction(command='C-c')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Then run the command again
        action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code == -1:
        # The previous command is still running
        # We need to kill previous command
        logger.info('The previous command is still running, trying to ctrl+z it...')
        action = CmdRunAction(command='C-z')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Then run the command again
        action = CmdRunAction(command=f'cd /workspace/{workspace_dir_name}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to cd to /workspace/{workspace_dir_name}: {str(obs)}',
    )

    action = CmdRunAction(command='git config --global core.pager ""')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git config --global core.pager "": {str(obs)}',
    )

    # First check for any git repositories in subdirectories
    action = CmdRunAction(command='find . -type d -name .git -not -path "./.git"')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to find git repositories: {str(obs)}',
    )

    git_dirs = [p for p in obs.content.strip().split('\n') if p]
    if git_dirs:
        # Remove all .git directories in subdirectories
        for git_dir in git_dirs:
            action = CmdRunAction(command=f'rm -rf "{git_dir}"')
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            assert_and_raise(
                isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
                f'Failed to remove git directory {git_dir}: {str(obs)}',
            )

    # add all files
    action = CmdRunAction(command='git add -A')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git add -A: {str(obs)}',
    )

    # Remove binary files from git staging
    action = CmdRunAction(command=remove_binary_files_from_git())
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to remove binary files: {str(obs)}',
    )

    n_retries = 0
    git_patch = None
    while n_retries < 5:
        action = CmdRunAction(
            command=f'git diff --no-color --cached {instance["base_commit"]} > patch.diff'
        )
        action.set_hard_timeout(max(300 + 100 * n_retries, 600))
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        n_retries += 1
        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                # Read the patch file
                action = FileReadAction(path='patch.diff')
                action.set_hard_timeout(max(300 + 100 * n_retries, 600))
                logger.info(action, extra={'msg_type': 'ACTION'})
                obs = runtime.run_action(action)
                logger.info(obs, extra={'msg_type': 'OBSERVATION'})
                if isinstance(obs, FileReadObservation):
                    git_patch = obs.content
                    break
                elif isinstance(obs, ErrorObservation):
                    # Fall back to cat "patch.diff" to get the patch
                    assert 'File could not be decoded as utf-8' in obs.content
                    action = CmdRunAction(command='cat patch.diff')
                    action.set_hard_timeout(max(300 + 100 * n_retries, 600))
                    logger.info(action, extra={'msg_type': 'ACTION'})
                    obs = runtime.run_action(action)
                    assert isinstance(obs, CmdOutputObservation) and obs.exit_code == 0
                    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
                    git_patch = obs.content
                    break
                else:
                    assert_and_raise(False, f'Unexpected observation type: {str(obs)}')
            else:
                logger.info('Failed to get git diff, retrying...')
                sleep_if_should_continue(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            sleep_if_should_continue(10)
        else:
            assert_and_raise(False, f'Unexpected observation type: {str(obs)}')

    assert_and_raise(git_patch is not None, 'Failed to get git diff (None)')

    # Remove binary diffs from the patch
    git_patch = remove_binary_diffs(git_patch)

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)
    return {'git_patch': git_patch}


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:

    pred_path = os.path.join(metadata.eval_output_dir, 'instances', instance.instance_id, "pred.jsonl")
    if not os.path.exists(pred_path):

        config = get_config(instance, metadata)

        # Setup the logger properly, so you can run multi-processing to parallelize the evaluation
        if reset_logger:
            log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
            reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
        else:
            logger.info(f'Starting evaluation for instance {instance.instance_id}.')

        # Increase resource_factor with increasing attempt_id
        if runtime_failure_count > 0:
            config.sandbox.remote_runtime_resource_factor = min(
                config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
                8,
            )
            logger.warning(
                f'This is the {runtime_failure_count + 1}th attempt for instance {instance.instance_id}, setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
            )

        metadata = copy.deepcopy(metadata)
        metadata.details['runtime_failure_count'] = runtime_failure_count
        metadata.details['remote_runtime_resource_factor'] = (
            config.sandbox.remote_runtime_resource_factor
        )

        runtime = create_runtime(config)
        call_async_from_sync(runtime.connect)

        try:
            initialize_runtime(runtime, instance, metadata)

            message_action = get_instruction(instance, metadata)

            # Here's how you can run the agent (similar to the `main` function) and get the final task state
            state: State | None = asyncio.run(
                run_controller(
                    config=config,
                    initial_user_action=message_action,
                    runtime=runtime,
                    fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                        metadata.agent_class
                    ],
                )
            )

            # if fatal error, throw EvalError to trigger re-run
            if is_fatal_evaluation_error(state.last_error):
                raise EvalException('Fatal error detected: ' + state.last_error)

            # ======= THIS IS SWE-Bench specific =======
            # Get git patch
            if DATASET_TYPE == 'SWE-bench-Live':
                from evaluation.benchmarks.swe_bench.live_utils import (
                    complete_runtime as complete_runtime_fn,
                )
            else:
                complete_runtime_fn = complete_runtime
            return_val = complete_runtime_fn(runtime, instance)
            git_patch = return_val['git_patch']
            logger.info(
                f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
            )
        finally:
            runtime.close()
        # ==========================================

        # ======= Attempt to evaluate the agent's edits =======
        # we use eval_infer.sh to evaluate the agent's edits, not here
        # because the agent may alter the environment / testcases
        test_result = {
            'git_patch': git_patch,
        }

        # If you are working on some simpler benchmark that only evaluates the final model output (e.g., in a MessageAction)
        # You can simply get the LAST `MessageAction` from the returned `state.history` and parse it for evaluation.
        if state is None:
            raise ValueError('State should not be None.')

        # NOTE: this is NO LONGER the event stream, but an agent history that includes delegate agent's events
        histories = [event_to_dict(event) for event in state.history]
        metrics = get_metrics(state)

        # Save the output
        instruction = message_action.content
        if message_action.image_urls:
            instruction += (
                '\n\n<image_urls>' + '\n'.join(message_action.image_urls) + '</image_urls>'
            )
        output = EvalOutput(
            instance_id=instance.instance_id,
            instruction=instruction,
            instance=instance.to_dict(),  # SWE Bench specific
            test_result=test_result,
            metadata=metadata,
            history=histories,
            metrics=metrics,
            error=state.last_error if state and state.last_error else None,
        )

        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        with open(pred_path, 'w') as output_fp:
            json.dump(output.model_dump(), output_fp)
    else:
        logger.info(f"{instance.instance_id} alreadly have pred.json file")
        output = instance

    run_eval(instance, pred_path, metadata.eval_output_dir, force_eval=metadata.details.get("force_eval", False))
    return output

def check_eval_finish(eval_dir):
    report_json = os.path.join(eval_dir, "report.json")
    report_log = os.path.join(eval_dir, "run_instance.log")
    if os.path.exists(report_json):
        return True
    elif os.path.exists(report_log):
        text = open(report_log).read()
        if "EvaluationError" in text and "Patch Apply Failed" in text:
            return True
    return False


def run_eval(instance, pred_path, output_dir, force_eval=False):
    report_dir = os.path.join(output_dir, "instances", instance.instance_id)
    if not force_eval and check_eval_finish(report_dir):
        logger.info(f"{instance.instance_id} alreadly have finished evaluation before")
        return

    jd = json.load(open(pred_path))
    pred = {
        'instance_id': jd['instance_id'],
        'model_name_or_path': jd['metadata']['llm_config']['model'],
        'model_patch': jd['test_result']['git_patch'],
    }
    if pred["model_patch"] == "":
        logger.info(f'instance {instance.instance_id} - EMPTY PATCH')
        return

    logger.info(f'Starting swebench evaluation for instance {pred['instance_id']}.')
    instance_id = pred['instance_id']
    namespace = DEFAULT_DOCKER_IMAGE_PREFIX.replace("docker.io/", "").rstrip("/")
    logger.info(f"{namespace}")
    instance_image_tag= "latest"

    run_instance(
        test_spec=make_test_spec(instance, namespace=namespace, instance_image_tag=instance_image_tag),
        pred=pred,
        rm_image=False,
        force_rebuild=False,
        client=docker.from_env(),
        run_id=instance_id,
        timeout=1800,
        rewrite_reports=False,
        log_dir=Path(report_dir),
    )

    return

def filter_dataset(dataset: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    valid_file_path = os.path.join(data_dir, "valid.txt")
    if os.path.exists(valid_file_path):
        valid_docker = set([a.strip() for a in open(valid_file_path)])

        logger.info(f"using valid.txt with for valid instance_id {len(valid_docker)}")
        dataset = dataset[dataset["instance_id"].isin(valid_docker)]

    print(f"valid subset: {len(dataset)}")
    return dataset

def prepare_dataset(
    dataset: pd.DataFrame,
    output_dir: str,
    eval_n_limit: int,
    eval_ids: list[str] | None = None,
    skip_num: int | None = None,
):
    assert 'instance_id' in dataset.columns, (
        "Expected 'instance_id' column in the dataset. You should define your own unique identifier for each instance and use it as the 'instance_id' column."
    )
    id_column = 'instance_id'
    logger.info(f'Writing evaluation output to {output_dir}')
    finished_ids: set[str] = set()
    eval_outputs_dir = os.path.join(output_dir, "instances")
    if os.path.exists(eval_outputs_dir):
        for instance_id in os.listdir(eval_outputs_dir):
            if check_eval_finish(os.path.join(eval_outputs_dir, instance_id)):
                finished_ids.add(instance_id)
    if eval_ids:
        eval_ids_converted = [dataset[id_column].dtype.type(id) for id in eval_ids]
        dataset = dataset[dataset[id_column].isin(eval_ids_converted)]
        logger.info(f'Limiting evaluation to {len(eval_ids)} specific instances.')
    elif eval_n_limit and eval_n_limit > 0:
        # Use fixed random seed 42 for sampling without replacement
        dataset = dataset.sample(
            min(eval_n_limit, len(dataset)), random_state=42, replace=False
        )
        logger.info(
            f'Randomly sampling {eval_n_limit} unique instances with random seed 42.'
        )

    def make_serializable(instance: pd.Series) -> dict:
        import numpy as np

        instance_dict = instance.to_dict()
        for k, v in instance_dict.items():
            if isinstance(v, np.ndarray):
                instance_dict[k] = v.tolist()
            elif isinstance(v, pd.Timestamp):
                instance_dict[k] = str(v)
        return instance_dict

    new_dataset = [
        make_serializable(instance)
        for _, instance in dataset.iterrows()
        if str(instance[id_column]) not in finished_ids
    ]
    logger.info(
        f'Finished instances: {len(finished_ids)}, Remaining instances: {len(new_dataset)}'
    )

    return pd.DataFrame(new_dataset)


def run_evaluation(
    dataset: pd.DataFrame,
    metadata: EvalMetadata | None,
    output_dir: str,
    num_workers: int,
    process_instance_func: Callable[
        [pd.Series, EvalMetadata, bool], Awaitable[EvalOutput]
    ],
    max_retries: int = 5,  # number of retries for each instance
    timeout_seconds: int | None = None,
):

    def update_progress(result: EvalOutput, pbar: tqdm):
        pbar.update(1)
        pbar.set_description(f'Instance {result.instance_id}')
        if "test_result" in result:
            pbar.set_postfix_str(f'Test Result: {str(result.test_result)[:300]}...')
            logger.info(f'Finished evaluation for instance {result.instance_id}: {str(result.test_result)[:300]}...\n')

    use_multiprocessing = num_workers > 1
    logger.info(
        f'Evaluation started with Agent {metadata.agent_class}:\n'
        f'model {metadata.llm_config.model}, max iterations {metadata.max_iterations}.\n'
    )

    total_instances = len(dataset)
    pbar = tqdm(total=total_instances, desc='Instances processed')

    try:
        if use_multiprocessing:
            with mp.Pool(num_workers) as pool:
                args_iter = (
                    (
                        process_instance_func,
                        instance,
                        metadata,
                        True,
                        max_retries,
                        timeout_seconds,
                    )
                    for _, instance in dataset.iterrows()
                )
                results = pool.imap_unordered(_process_instance_wrapper_mp, args_iter)
                for result in results:
                    update_progress(result, pbar)
        else:
            for _, instance in dataset.iterrows():
                result = _process_instance_wrapper(
                    process_instance_func=process_instance_func,
                    instance=instance,
                    metadata=metadata,
                    use_mp=False,
                    max_retries=max_retries,
                )
                update_progress(result, pbar)

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt received. Cleaning up...\n')
        cleanup()

    # Check if any instances reached maximum retries
    if metadata and metadata.eval_output_dir:
        check_maximum_retries_exceeded(metadata.eval_output_dir)

SKIP = []
def load_dataset_local(data_path, split):
    data_dir = os.path.join(data_path, "split", split)
    assert os.path.exists(data_dir), f"{data_dir} not exist"

    jd_list = []
    for fname in os.listdir(data_dir):
        if fname in SKIP:
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            jd = json.load(f)
            jd["data_json_path"] = fpath
            if "PASS_TO_PASS" in jd:
                jd["PASS_TO_PASS"] = json.dumps(jd["PASS_TO_PASS"])
            if "FAIL_TO_PASS" in jd:
                jd["FAIL_TO_PASS"] = json.dumps(jd["FAIL_TO_PASS"])
            jd_list.append(jd)

    df = pd.DataFrame(jd_list)
    return df


if __name__ == '__main__':
    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='princeton-nlp/SWE-bench',
        help='data set to evaluate on, either full-test or lite-test',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='split to evaluate on',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='swe',
        choices=['swe', 'swt', 'swt-ci'],
        help="mode to run the evaluation, either 'swe', 'swt', or 'swt-ci'",
    )
    parser.add_argument('--force_eval', action="store_true")

    args, _ = parser.parse_known_args()

    args.agent_cls = "CodeActAgent"
    args.mode = "swe"
    args.eval_note = "v2-distilling"

    args.dataset = "SWE-bench-Rebench"
    args.max_iterations = 100
    args.eval_num_workers = 1
    args.split = "test"
    args.eval_n_limit = 1
    args.llm_config = "llm.deepswe32b_cyber3"

    os.environ["EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED"] = "true"

    args.dataset_dir = "/shared_workspace/yanruo/data/Public/SWE-rebench"
    dataset = load_dataset_local(args.dataset_dir, args.split)
    # dataset = load_dataset(args.dataset, split=args.split)

    set_dataset_type(args.dataset)

    swe_bench_tests = filter_dataset(dataset, args.dataset_dir)
    logger.info(f'Loaded dataset {args.dataset} with split {args.split}: {len(swe_bench_tests)} tasks')


    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config, args.config_file)
        llm_config.log_completions = True
        # modify_params must be False for evaluation purpose, for reproducibility and accurancy of results
        llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    agent_config = None
    if args.agent_config:
        agent_config = get_agent_config_arg(args.agent_config, args.config_file)

    details = {'mode': args.mode, "force_eval": args.force_eval}
    dataset_descrption = (args.dataset.replace('/', '__') + '-' + args.split.replace('/', '__'))
    metadata = make_metadata(
        llm_config,
        dataset_descrption,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
        condenser_config=NoOpCondenserConfig(),
    )

    output_dir = metadata.eval_output_dir

    # load the dataset
    instances = prepare_dataset(swe_bench_tests, output_dir, args.eval_n_limit)
    run_evaluation(
        instances,
        metadata,
        output_dir,
        args.eval_num_workers,
        process_instance,
        timeout_seconds=4 * 60 * 60,  # 8 hour PER instance should be more than enough
        max_retries=3,
        )
