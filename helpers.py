
import os.path
import pathlib
import subprocess
import datetime
import tempfile


def step(*args):
    ''' Prints steps '''
    status = '[' + str(datetime.datetime.now().strftime("%H:%M:%S")) + '] '
    print(status + ' '.join(map(str, args)))


def do_cmd(*args):
    step(' '.join(map(str, args)))
    stdout, stderr = subprocess.Popen(
        map(str, args), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE).communicate()
    print(stdout.decode("utf-8"))
    print(stderr.decode("utf-8"))
    return stdout, stderr


def create_tmpdir(tmpdir):
    ''' Creates tmp dir. Defaults to current_dir/tmp '''
    return pathlib.Path(tempfile.mkdtemp(dir=str(tmpdir)))

