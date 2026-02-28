"""Debug: print the exact prompt sent to the model for 2k, 10u."""
import sys, random
sys.path.insert(0, '.')
from core.dataset_configs import format_for_chat, SYSTEM_PROMPT, ORIGINAL_CATEGORIES
from core.model_loader import verify_single_token
from core.model_loader import load_model

model, tokenizer, info = load_model('Qwen/Qwen2.5-0.5B-Instruct', n_ctx=2048)
value_to_tid = verify_single_token(tokenizer)
value_pool = list(value_to_tid.keys())

num_keys = 2
num_updates = 10
condition = 'PI'
seed = hash(('PI', 0, 10, 2, 'recal')) % (2**31)

rng = random.Random(seed)
categories = rng.sample(ORIGINAL_CATEGORIES, num_keys)
total_needed = num_keys * num_updates
selected = rng.sample(value_pool, total_needed)
values_per_cat = {}
idx = 0
for cat in categories:
    values_per_cat[cat] = selected[idx:idx + num_updates]
    idx += num_updates

test_cat = categories[seed % num_keys]

items = []
for cat in categories:
    for val in values_per_cat[cat]:
        items.append({'category': cat, 'value': val})

rng.shuffle(items)
for _ in range(100):
    ok = all(items[i]['category'] != items[i-1]['category'] for i in range(1, len(items)))
    if ok:
        break
    rng.shuffle(items)

stream_lines = [f"{it['category']}: {it['value']}" for it in items]
stream_text = '\n'.join(stream_lines)
query_word = 'first' if condition == 'RI' else 'last'
cat_values = [it['value'] for it in items if it['category'] == test_cat]
expected = cat_values[0] if condition == 'RI' else cat_values[-1]

prompt = (
    f'Read the following key-value stream. Each key gets updated multiple times.\n\n'
    f'{stream_text}\n\n'
    f'What was the {query_word} value of {test_cat}?'
)

print('=== RAW PROMPT ===')
print(prompt)
print()
print(f'=== EXPECTED ANSWER: {expected} ===')
print(f'=== TEST CATEGORY: {test_cat} ===')
print(f'=== CATEGORIES: {categories} ===')
print()

formatted = format_for_chat(prompt, tokenizer)
print('=== FULL FORMATTED PROMPT (sent to model) ===')
print(formatted)
print()
print(f'=== TOKEN COUNT: {len(tokenizer.encode(formatted))} ===')
