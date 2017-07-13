import planingfsi.config as config

print

for key in config.__dict__:
    if not key.startswith('_'):
        print '{0} = {1}'.format(key, getattr(config, key))
