#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import os
import asyncio
import shutil
import yaml
import re
import fnmatch
import shlex
import configparser
import pathlib

from typing import List, Dict, Any, Optional, Union, NamedTuple, Set

from . import utils

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LINT_YML = REPO_ROOT / ".github" / "workflows" / "lint.yml"


class col:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def should_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def color(the_color: str, text: str) -> str:
    if should_color():
        return col.BOLD + the_color + str(text) + col.RESET
    else:
        return text


def cprint(the_color: str, text: str) -> None:
    if should_color():
        print(color(the_color, text))
    else:
        print(text)



def git(args: List[str]) -> List[str]:
    p = subprocess.run(
        ["git"] + args,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    lines = p.stdout.decode().strip().split("\n")
    return [line.strip() for line in lines]


def find_changed_files() -> List[str]:
    untracked = []

    for line in git(["status", "--porcelain"]):
        # Untracked files start with ??, so grab all of those
        if line.startswith("?? "):
            untracked.append(line.replace("?? ", ""))

    # Modified, unstaged
    modified = git(["diff", "--name-only"])

    # Modified, staged
    cached = git(["diff", "--cached", "--name-only"])

    # Committed
    merge_base = git(["merge-base", "origin/master", "HEAD"])[0]
    diff_with_origin = git(["diff", "--name-only", merge_base, "HEAD"])

    # De-duplicate
    all_files = set()
    for x in untracked + cached + modified + diff_with_origin:
        stripped = x.strip()
        if stripped != "" and os.path.exists(stripped):
            all_files.add(stripped)
    return list(all_files)


def print_results(job_name: str, passed: bool, streams: List[str]) -> None:
    icon = color(col.GREEN, "✓") if passed else color(col.RED, "x")
    print(f"{icon} {color(col.BLUE, job_name)}")

    for stream in streams:
        stream = stream.strip()
        if stream != "":
            print(stream)


class CommandResult(NamedTuple):
    passed: bool
    stdout: str
    stderr: str

    @staticmethod
    def combine_streams(a: str, b: str) -> str:
        if a.strip() == "" and b.strip() == "":
            return ""
        elif a.strip() == "":
            return b
        elif b.strip() == "":
            return a

        return a + "\n" + b

    def __add__(self, other: "CommandResult") -> "CommandResult":  # type: ignore[override]
        return CommandResult(
            self.passed and other.passed,
            self.stdout + "\n" + other.stdout,
            self.stderr + "\n" + other.stderr,
        )


async def shell_cmd(
    cmd: Union[str, List[str]],
    env: Optional[Dict[str, Any]] = None,
    redirect: bool = True,
    check: bool = False,
) -> CommandResult:
    if isinstance(cmd, list):
        cmd_str = " ".join(shlex.quote(arg) for arg in cmd)
    else:
        cmd_str = cmd

    utils.log("Running", cmd_str)
    proc = await asyncio.create_subprocess_shell(
        cmd_str,
        shell=True,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE if redirect else None,
        stderr=subprocess.PIPE if redirect else None,
        executable=shutil.which("bash"),
    )
    stdout, stderr = await proc.communicate()

    passed = proc.returncode == 0
    if not redirect:
        return CommandResult(passed, "", "")

    stdout = stdout.decode().strip()
    stderr = stderr.decode().strip()

    if check and not passed:
        raise RuntimeError(f"'{cmd_str}' failed:\n{stdout}\n{stderr}")

    return CommandResult(passed, stdout, stderr)


class Check:
    name: str

    async def pre_run(self) -> None:
        pass

    async def run(self, files: Optional[List[str]]) -> bool:
        await self.pre_run()
        result = await self.run_helper(files)
        if result is None:
            return True

        streams = []
        if not result.passed:
            streams = [
                result.stderr,
                result.stdout,
            ]
        print_results(self.name, result.passed, streams)
        return result.passed

    async def run_helper(self, files: Optional[List[str]]) -> Optional[CommandResult]:
        if files is not None:
            relevant_files = self.filter_files(files)
            if len(relevant_files) == 0:
                # No files, do nothing
                return CommandResult(passed=True, stdout="", stderr="")

            return await self.quick(relevant_files)

        return await self.full()

    def filter_ext(self, files: List[str], extensions: Set[str]) -> List[str]:
        def passes(filename: str) -> bool:
            return os.path.splitext(filename)[1] in extensions

        return [f for f in files if passes(f)]

    def filter_files(self, files: List[str]) -> List[str]:
        return files

    async def quick(self, files: List[str]) -> CommandResult:
        raise NotImplementedError

    async def full(self) -> Optional[CommandResult]:
        raise NotImplementedError


class Flake8(Check):
    name = "flake8"
    config_path = REPO_ROOT / ".flake8"
    common_cmd = ["flake8", "--config", str(config_path)]

    def filter_files(self, files: List[str]) -> List[str]:
        config = configparser.ConfigParser()
        config.read(self.config_path)

        excludes = re.split(r",\s*", config["flake8"]["exclude"].strip())
        excludes = [e.strip() for e in excludes if e.strip() != ""]

        def should_include(name: str) -> bool:
            for exclude in excludes:
                if fnmatch.fnmatch(name, pat=exclude):
                    return False
                if name.startswith(exclude) or f"./{name}".startswith(exclude):
                    return False
            return True

        files = self.filter_ext(files, {".py"})
        return [f for f in files if should_include(f)]

    async def quick(self, files: List[str]) -> CommandResult:
        return await shell_cmd(self.common_cmd + files)

    async def full(self) -> CommandResult:
        return await shell_cmd(self.common_cmd)


class Mypy(Check):
    name = "mypy"

    def __init__(self, generate_stubs: bool):
        self.generate_stubs = generate_stubs

    def filter_files(self, files: List[str]) -> List[str]:
        return self.filter_ext(files, {".py", ".pyi"})

    def env(self) -> Dict[str, Any]:
        env = os.environ.copy()
        if should_color():
            # Secret env variable: https://github.com/python/mypy/issues/7771
            env["MYPY_FORCE_COLOR"] = "1"
        return env

    async def quick(self, files: List[str]) -> CommandResult:
        if self.generate_stubs:
            await self.autogen()

        result = await shell_cmd(
            [sys.executable, "tools/linter/mypy/mypy_wrapper.py"]
            + [os.path.join(REPO_ROOT, f) for f in files],
            env=self.env(),
        )
        return result

    async def run(self, files: Optional[List[str]]) -> bool:
        passed = await super().run(files)
        if not passed:
            print("Mypy failed, you may need to run again with --generate-stubs")

        return passed

    async def autogen(self) -> CommandResult:
        time = shutil.which("time")
        if time is None:
            raise RuntimeError("Unable to find 'time' executable")
        result = CommandResult(True, "", "")
        coros = [
            shell_cmd(
                [
                    sys.executable,
                    "-m",
                    "tools.generate_torch_version",
                    "--is_debug=false",
                ]
            ),
            shell_cmd(
                [
                    sys.executable,
                    "-m",
                    "tools.codegen.gen",
                    "-s",
                    "aten/src/ATen",
                    "-d",
                    "build/aten/src/ATen",
                ]
            ),
            shell_cmd(
                [
                    sys.executable,
                    "-m",
                    "tools.pyi.gen_pyi",
                    "--native-functions-path",
                    "aten/src/ATen/native/native_functions.yaml",
                    "--deprecated-functions-path",
                    "tools/autograd/deprecated.yaml",
                ]
            ),
        ]
        results = await utils.gather(coros)
        result = CommandResult(True, "", "")
        for r in results:
            result += r
        # result += await shell_cmd(
        #     [sys.executable, "-m", "tools.generate_torch_version", "--is_debug=false"]
        # )

        # result += await

        # result += await shell_cmd(
        #     [
        #         sys.executable,
        #         "-m",
        #         "tools.pyi.gen_pyi",
        #         "--native-functions-path",
        #         "aten/src/ATen/native/native_functions.yaml",
        #         "--deprecated-functions-path",
        #         "tools/autograd/deprecated.yaml",
        #     ]
        # )

        return CommandResult(*result)

    async def full(self) -> CommandResult:
        if self.generate_stubs:
            result = await self.autogen()
        else:
            result = CommandResult(True, "", "")

        exe = shutil.which("mypy")
        if exe is None:
            raise RuntimeError("Unable to find 'mypy' executable")

        coros = [
            shell_cmd([exe, "--config", str(config)], env=self.env())
            for config in REPO_ROOT.glob("mypy*.ini")
        ]

        results = await utils.gather(coros)
        for r in results:
            if isinstance(r, BaseException):
                raise r
            result += r

        return CommandResult(*result)


class ShellCheck(Check):
    name = "shellcheck: Run ShellCheck"

    def filter_files(self, files: List[str]) -> List[str]:
        return self.filter_ext(files, {".sh"})

    async def quick(self, files: List[str]) -> CommandResult:
        return await shell_cmd(
            ["tools/linter/run_shellcheck.sh"]
            + [os.path.join(REPO_ROOT, f) for f in files],
        )

    async def full(self) -> CommandResult:
        return await shell_cmd(
            [
                sys.executable,
                "tools/actions_local_runner.py",
                "--job",
                "shellcheck",
                "--file",
                ".github/workflows/lint.yml",
                "--step",
                "Run ShellCheck",
            ],
            redirect=False,
        )


class ClangTidy(Check):
    name = "clang-tidy"
    # common_options = [
    #     "--clang-tidy-exe",
    #     ".clang-tidy-bin/clang-tidy",
    # ]
    def __init__(self, exe: str, generate_build: bool):
        self.exe = exe
        self.generate_build = generate_build
        self.common_options = [
            "--clang-tidy-exe",
            str(exe),
        ]

    # if not pathlib.Path("build").exists():
    #     generate_build_files()

    # # Check if clang-tidy executable exists
    # exists = os.access(options.clang_tidy_exe, os.X_OK)

    # if not exists:
    #     msg = (
    #         f"Could not find '{options.clang_tidy_exe}'\n"
    #         + "We provide a custom build of clang-tidy that has additional checks.\n"
    #         + "You can install it by running:\n"
    #         + "$ python3 tools/linter/install/clang_tidy.py"
    #     )
    #     raise RuntimeError(msg)

    # result, _ = run(options)
    async def pre_run(self):
        if self.generate_build:
            await self.update_submodules()
            await self.gen_compile_commands()
            await self.run_autogen()
            
    async def update_submodules(self) -> CommandResult:
        return await shell_cmd(["git", "submodule", "update", "--init", "--recursive"])


    async def gen_compile_commands(self) -> CommandResult:
        os.environ["USE_NCCL"] = "0"
        os.environ["USE_DEPLOY"] = "1"
        os.environ["CC"] = "clang"
        os.environ["CXX"] = "clang++"
        return await shell_cmd(["time", sys.executable, "setup.py", "--cmake-only", "build"])


    async def run_autogen(self) -> CommandResult:
        result = await shell_cmd(
            [
                "time",
                sys.executable,
                "-m",
                "tools.codegen.gen",
                "-s",
                "aten/src/ATen",
                "-d",
                "build/aten/src/ATen",
            ]
        )

        result += await shell_cmd(
            [
                "time",
                sys.executable,
                "tools/setup_helpers/generate_code.py",
                "--declarations-path",
                "build/aten/src/ATen/Declarations.yaml",
                "--native-functions-path",
                "aten/src/ATen/native/native_functions.yaml",
                "--nn-path",
                "aten/src",
            ]
        )

        return result


    def filter_files(self, files: List[str]) -> List[str]:
        return self.filter_ext(files, {".c", ".cc", ".cpp"})

    async def quick(self, files: List[str]) -> CommandResult:
        return await shell_cmd(
            [sys.executable, "-m", "tools.linter.clang_tidy", "--paths"]
            + [os.path.join(REPO_ROOT, f) for f in files]
            + self.common_options,
        )

    async def full(self) -> CommandResult:
        # from .clang_tidy.run import _run_clang_tidy
        # from .clang_tidy import run
        # await _run_clang_tidy(None, [], [])
        return await shell_cmd(
            [sys.executable, "-m", "tools.linter.clang_tidy"] + self.common_options,
            redirect=False,
        )


class YamlStep(Check):
    def __init__(self, step: Dict[str, Any], job_name: str):
        self.step = step
        self.name = f'{job_name}: {self.step["name"]}'

    async def quick(self, files: List[str]) -> CommandResult:
        return await self.full()

    async def full(self) -> CommandResult:
        env = os.environ.copy()
        env["GITHUB_WORKSPACE"] = "/tmp"
        script = self.step["run"]

        if utils.VERBOSE:
            # TODO: Either lint that GHA scripts only use 'set -eux' or make this more
            # resilient
            script = script.replace("set -eux", "set -eu")
            script = re.sub(r"^time ", "", script, flags=re.MULTILINE)

        return await shell_cmd(script, env=env)


class QuickChecks(YamlStep):
    def __init__(self):
        steps = lint_yaml_steps("quick-checks", ["Ensure no trailing spaces"])
        print(steps)
        super().__init__()



def changed_files() -> Optional[List[str]]:
    changed_files: Optional[List[str]] = None
    try:
        changed_files = sorted(find_changed_files())
    except Exception:
        # If the git commands failed for some reason, bail out and use the whole list
        print(
            "Could not query git for changed files, falling back to testing all files instead",
            file=sys.stderr,
        )
        return None

    return changed_files


def grab_specific_steps(
    steps_to_grab: List[str], job: Dict[str, Any]
) -> List[Dict[str, Any]]:
    relevant_steps = []
    for step in steps_to_grab:
        for actual_step in job["steps"]:
            if actual_step["name"].lower().strip() == step.lower().strip():
                relevant_steps.append(actual_step)
                break

    if len(relevant_steps) != len(steps_to_grab):
        raise RuntimeError(f"Missing steps:\n{relevant_steps}\n{steps_to_grab}")

    return relevant_steps


def extract_step(
    step: str, job: Dict[str, Any]
) -> List[Dict[str, Any]]:
    relevant_steps = []
    for actual_step in job["steps"]:
        if actual_step["name"].lower().strip() == step.lower().strip():
            return actual_step

    raise RuntimeError(f"Missing step:\n{step}")


def lint_yaml_steps(job: str, steps: List[str]):
    with open(LINT_YML) as f:
        action = yaml.safe_load(f)

    return grab_specific_steps(steps, action["jobs"][job])


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Pull shell scripts out of GitHub actions and run them"
#     )
#     parser.add_argument("--file", help="YAML file with actions")
#     parser.add_argument(
#         "--changed-only",
#         help="only run on changed files",
#         action="store_true",
#         default=False,
#     )
#     parser.add_argument("--job", help="job name", required=True)
#     parser.add_argument(
#         "--no-quiet", help="output commands", action="store_true", default=False
#     )
#     parser.add_argument("--step", action="append", help="steps to run (in order)")
#     args = parser.parse_args()

#     quiet = not args.no_quiet

#     if args.file is None:
#         # If there is no .yml file provided, fall back to the list of known
#         # jobs. We use this for flake8 and mypy since they run different
#         # locally than in CI due to 'make quicklint'
#         if args.job not in ad_hoc_steps:
#             raise RuntimeError(
#                 f"Job {args.job} not found and no .yml file was provided"
#             )

#         files = None
#         if args.changed_only:
#             files = changed_files()

#         checks = [ad_hoc_steps[args.job](files, quiet)]
#     else:
#         if args.step is None:
#             raise RuntimeError("1+ --steps must be provided")

#         action = yaml.safe_load(open(args.file, "r"))
#         if "jobs" not in action:
#             raise RuntimeError(f"top level key 'jobs' not found in {args.file}")
#         jobs = action["jobs"]

#         if args.job not in jobs:
#             raise RuntimeError(f"job '{args.job}' not found in {args.file}")

#         job = jobs[args.job]

#         # Pull the relevant sections out of the provided .yml file and run them
#         relevant_steps = grab_specific_steps(args.step, job)
#         checks = [
#             YamlStep(step=step, job_name=args.job, quiet=quiet)
#             for step in relevant_steps
#         ]

#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(asyncio.gather(*[check.run() for check in checks]))


# # These are run differently locally in order to enable quicklint, so dispatch
# # out to special handlers instead of using lint.yml
# ad_hoc_steps = {
#     "mypy": Mypy,
#     "flake8-py3": Flake8,
#     "shellcheck": ShellCheck,
#     "clang-tidy": ClangTidy,
# }

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         pass
