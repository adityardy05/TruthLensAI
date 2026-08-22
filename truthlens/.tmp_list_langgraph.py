import importlib, pkgutil

print('Python:', importlib.util.find_spec('sys').name)
try:
    mod = importlib.import_module('langgraph.checkpoint')
    print('checkpoint __path__:', getattr(mod, '__path__', None))
    subs = [m.name for m in pkgutil.iter_modules(mod.__path__)]
    print('submodules:', subs)
except Exception as e:
    print('checkpoint import error:', e)

try:
    im = importlib.import_module('langgraph.checkpoint.sqlite')
    print('imported langgraph.checkpoint.sqlite OK, attrs:', [a for a in dir(im) if a.lower().startswith('sql') or 'sqlite' in a.lower() or 'save' in a.lower()][:100])
except Exception as e:
    print('sqlite import error:', e)
