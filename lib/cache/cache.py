import hashlib
import msgpack
import os
import shutil
from functools import wraps
from queue import Queue
import lib.config as config


"""
Cache which can wrap functions to automatically cache
results based on the parameters.
"""


cache_dir = config.CACHE_PATH
disable = False
update = False
cache_in_memory = config.CACHE_IN_MEMORY
cache_files = None


def set_cache_dir(new_dir):
    global cache_dir, cache_files
    cache_dir = new_dir
    # reset file cache
    cache_files = None


def set_disable_cache(disable_cache):
    global disable
    disable = disable_cache


def set_update_cache(update_cache):
    global update
    update = update_cache


def hash_f(args):
    """Default hash function for caching."""
    # Only hash arguments which are too long. Otherwise use exact string.
    res = ''
    first = True
    for arg in args:
        if not first:
            res += '_'
        first = False
        val = str(arg).replace('_', '-').replace('/', '-')
        if len(val) > 32:
            val = hashlib.md5(val.encode('utf8')).hexdigest()
        res += val
    return res


def cache(*args, hash_f=hash_f, ignore_args=[], default=None,
          object_hook=None, in_memory=False, priority=-1):
    """Decorator function which uses the args as the file name"""
    other_args = list(args)

    # Create decorator
    def decorator(f):
        @wraps(f)
        def wrapper(*args):
            if disable:
                return f(*args)
            all_args = []
            all_args.extend(other_args)
            for idx, arg in enumerate(args):
                if idx not in ignore_args:
                    all_args.append(arg)
            file_id = f.__name__ + '/' + hash_f(all_args)
            path = get_path(file_id)
            if not is_cached(path) or update:
                res = f(*args)
                dump(path, res, default=default,
                     in_memory=in_memory, priority=priority)
                return res
            return load(path, object_hook=object_hook, priority=priority)
        return wrapper
    return decorator


def cache_load(*args, hash_f=hash_f, **kwargs):
    """Load from the cache based on the args given"""
    if disable:
        return None
    file_id = args[0] + '/' + hash_f(args[1:])
    path = get_path(file_id)
    return load(path, **kwargs)


def cache_store(*args, hash_f=hash_f, **kwargs):
    """Store into the cache based on the args given"""
    if disable:
        return
    file_id = args[0] + '/' + hash_f(args[1:-1])
    path = get_path(file_id)
    res = args[-1]
    dump(path, res, **kwargs)


def get_path(file_id):
    """Utility to get path in cache_dir"""
    return cache_dir + '/' + file_id


def read_cache_files():
    """Read the files on disk in advance."""
    global cache_files
    cache_files = set()
    for root, subdirs, files in os.walk(cache_dir):
        for file in files:
            cache_files.add(root + '/' + file)


def is_cached(file, on_disk=False, in_memory=False):
    """Check if file is cached"""
    # Read cache files if not read already
    if cache_files is None:
        read_cache_files()

    # Check if cached
    if not on_disk and not in_memory:
        return file in cache_files or file in loaded_cache
    if on_disk:
        return file in cache_files
    if in_memory:
        return file in loaded_cache


# Used to keep data that is recently used already fetched off the disk
# This doesn't actually fix the spacy issue, which is why there is now
# support to cache objects in memory even if they are never written to disk
loaded_cache = {}
add_order = Queue()
add_count = {}
total_added = 0


def add_to_loaded(file, res, priority=None):
    """Add object to currently loaded objects in memory"""
    global loaded_cache, add_order, add_count, total_added
    loaded_cache[file] = res
    # Currently priority=0 will force values to be kept in memory always
    # Eventually I can add a more in depth priority-system
    if priority != 0:
        add_order.put(file)
        if file not in add_count:
            total_added += 1
        add_count[file] = add_count.get(file, 0) + 1
        if total_added > config.IN_MEMORY_CACHE_SIZE:
            remove_from_loaded()


def remove_from_loaded():
    """
    Remove oldest object in the memory.
    """
    global loaded_cache, add_order, add_count, total_added
    while not add_order.empty():
        file = add_order.get()
        assert(add_count[file] > 0)
        if add_count[file] == 1:
            total_added -= 1
            del loaded_cache[file]
            del add_count[file]
            break
        else:
            add_count[file] = add_count[file] - 1


def load(file, object_hook=None, priority=None):
    """Load cached result"""
    if cache_in_memory and is_cached(file, in_memory=True):
        res = loaded_cache[file]
        return res
    if is_cached(file, on_disk=True):
        with open(file, 'rb') as f:
            if object_hook is None:
                res = msgpack.unpackb(f.read(), use_list=False, raw=False)
            else:
                res = msgpack.unpackb(
                    f.read(), object_hook=object_hook,
                    use_list=False, raw=False)
        if cache_in_memory:
            add_to_loaded(file, res, priority=priority)
        return res
    return None


def dump(file, val, default=None, in_memory=False, priority=None):
    """Dump a cached result"""
    if cache_in_memory:
        add_to_loaded(file, val, priority=None)
    # cache to disk if not storing only in memory
    if not in_memory:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        cache_files.add(file)
        with open(file, 'wb') as f:
            if default is None:
                f.write(msgpack.packb(val))
            else:
                f.write(msgpack.packb(val, default=default))


def reset(f):
    """Delete the cache for the function f"""
    name = f.__name__
    shutil.rmtree(get_path(name))
