import json
import os

file = open('reddit_casual.json')
result_file = open('reddit_casual_transformed.json', 'w', encoding='utf-8')

data = json.load(file)

transformed_data = []

for conversation in data:
    text = ''
    current_char = 1 - conversation['lines'][0]['character']
    for line in conversation['lines']:
        if line['character'] != current_char:
            if text != '':
                text += '\n'
            text += f"{'A' if line['character'] == 0 else 'B'}: {line['text']}"
        else:
            text += f"\n{line['text']}"
        current_char = line['character']
    transformed_data.append({'text': text})

result_file.write(json.dumps(transformed_data))

