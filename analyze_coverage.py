import json

with open('coverage.json') as f:
    data = json.load(f)

print('Total Coverage:', data['totals']['percent_covered'], '%')
print('\nModules with low coverage (<70%):')
files = data['files']
low_cov = [(f, files[f]['summary']['percent_covered']) for f in files if files[f]['summary']['percent_covered'] < 70]
low_cov.sort(key=lambda x: x[1])

for f, cov in low_cov:
    module_name = f.split('/')[-1]
    print(f'{cov:5.1f}% - {module_name}')
