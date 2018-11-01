
import re


# a decorator to add a 'static' variable within a Python function
def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


file_size = 0

@static_var("counter", 0)
def progress_fn(stream = None, chunk = None, file_handle = None, remaining = None):
    try:
        global file_size
        _progress_fn.counter = _progress_fn.counter + 1
        if _progress_fn.counter > 20:         # avoid too many repeated calls to the callback function
            percent = (100 * (file_size - remaining)) / file_size       # percentage of the file that has been downloaded
            print("\r{:00.0f}% downloaded".format(percent), end='')
            _progress_fn.counter = 0
    except:
        pass    # not critical


# on_complete_callback takes 2 parameters
def complete_fn(stream = None, file_handle = None,):
    print("\n")


_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]  
