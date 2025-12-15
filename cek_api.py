import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
genai.configure(api_key=api_key)

print("=" * 80)
print("ALL 53 AVAILABLE MODELS")
print("=" * 80)

models_list = []

for idx, m in enumerate(genai.list_models(), 1):
    model_name = m.name. replace('models/', '')
    print(f"\n{idx}.  {model_name}")
    print(f"   Display:  {m.display_name}")
    print(f"   Methods: {', '.join(m.supported_generation_methods)}")
    print(f"   Description: {m. description[: 100]}...")
    
    models_list.append({
        'name': model_name,
        'full_name': m.name,
        'methods': m.supported_generation_methods,
        'description': m. description
    })

print("\n" + "=" * 80)
print("MODELS WITH 'generateContent' SUPPORT")
print("=" * 80)

generate_content_models = [m for m in models_list if 'generateContent' in m['methods']]

print(f"\nFound {len(generate_content_models)} models with generateContent support:\n")

for m in generate_content_models: 
    print(f"✅ {m['name']}")
    print(f"   Full:  {m['full_name']}")
    desc = m['description']. lower()
    if any(word in desc for word in ['vision', 'image', 'multimodal', 'visual']):
        print(f"   👁️  LIKELY SUPPORTS VISION")
    print()

print("=" * 80)
print("Copy the output above and send it to me!")
print("=" * 80)