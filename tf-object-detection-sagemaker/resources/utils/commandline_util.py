import os
import sys
import subprocess


def run(cmd):
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=os.environ,
                               universal_newlines=True,
                               bufsize=1)

    while process.poll() is None:
        for stdout_line in process.stdout:
            print(stdout_line.strip(), flush=True)

    # stdout, stderr = process.communicate()

    stdout_line = process.communicate()[0]

    print(stdout_line.strip(), flush=True)

    return_code = process.poll()

    if return_code:
        error_msg = 'Return Code: {}, CMD: {}'.format(return_code, cmd)
        raise Exception(error_msg)


def run_python_script(script_name, params):
    python_executable = sys.executable

    script_cmd = [python_executable, script_name] + params
    run(script_cmd)


def run_command(command, params):
    command_cmd = [command] + params
    run(command_cmd)
