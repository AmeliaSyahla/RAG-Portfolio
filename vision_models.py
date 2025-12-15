import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Load API Key
api_key = os.environ.get("GEMINI_API_KEY", "").strip()

if not api_key:
    print("❌ GEMINI_API_KEY not found in environment!")
    print("Please add it to your .env file:")
    print("GEMINI_API_KEY=your_api_key_here")
    exit(1)

print(f"✅ API Key loaded (length: {len(api_key)})")
print(f"   Prefix: {api_key[: 20]}...")
print()

# Configure Gemini
try:
    genai.configure(api_key=api_key)
    print("✅ Gemini API configured successfully")
except Exception as e:
    print(f"❌ Failed to configure Gemini API: {str(e)}")
    exit(1)

print("\n" + "=" * 80)
print("ALL AVAILABLE GEMINI MODELS")
print("=" * 80)

try:
    models = genai.list_models()
    all_models = list(models)
    
    if not all_models:
        print("⚠️ No models found!")
    else:
        print(f"\nTotal models found: {len(all_models)}\n")
        
        vision_models = []
        text_models = []
        
        for idx, model in enumerate(all_models, 1):
            print(f"\n{'='*80}")
            print(f"MODEL #{idx}")
            print(f"{'='*80}")
            print(f"Name: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Description:  {model.description}")
            print(f"Supported Methods: {', '.join(model.supported_generation_methods)}")
            
            # Check for vision capability
            is_vision = False
            if hasattr(model, 'supported_generation_methods'):
                if 'generateContent' in model.supported_generation_methods:
                    # Check description for vision keywords
                    desc_lower = model.description.lower()
                    name_lower = model.name.lower()
                    
                    if any(keyword in desc_lower or keyword in name_lower 
                           for keyword in ['vision', 'image', 'multimodal', 'visual']):
                        is_vision = True
                        print(f"👁️  VISION CAPABLE:  YES")
                        vision_models.append(model.name)
                    else:
                        text_models.append(model.name)
            
            # Additional model info if available
            if hasattr(model, 'input_token_limit'):
                print(f"Input Token Limit: {model.input_token_limit}")
            if hasattr(model, 'output_token_limit'):
                print(f"Output Token Limit: {model.output_token_limit}")
            if hasattr(model, 'temperature'):
                print(f"Temperature: {model.temperature}")
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Total Models: {len(all_models)}")
        print(f"👁️  Vision Models: {len(vision_models)}")
        print(f"📝 Text Models: {len(text_models)}")
        
        if vision_models:
            print("\n" + "=" * 80)
            print("VISION-CAPABLE MODELS (USE THESE FOR IMAGES)")
            print("=" * 80)
            for vm in vision_models:
                clean_name = vm.replace('models/', '')
                print(f"  ✅ {clean_name}")
        
        if text_models:
            print("\n" + "=" * 80)
            print("TEXT-ONLY MODELS")
            print("=" * 80)
            for tm in text_models[: 5]:  # Show first 5
                clean_name = tm.replace('models/', '')
                print(f"  📝 {clean_name}")
            if len(text_models) > 5:
                print(f"  ... and {len(text_models) - 5} more")

except Exception as e:
    print(f"\n❌ Error listing models:  {str(e)}")
    print(f"\nFull error details:")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TESTING VISION MODELS")
print("=" * 80)

# Test vision models with a simple image
from PIL import Image

# Create a simple test image
test_image = Image.new('RGB', (100, 100), color='blue')

common_vision_models = [
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemini-pro-vision',
    'gemini-1.5-flash-8b',
    'gemini-1.5-flash-latest',
    'gemini-1.5-pro-latest',
    'models/gemini-1.5-flash',
    'models/gemini-1.5-pro',
    'models/gemini-pro-vision',
]

working_models = []

for model_name in common_vision_models: 
    try:
        print(f"\n🔄 Testing: {model_name}")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            ["What color is this image?  Answer in one word.", test_image],
            generation_config={'temperature': 0.1, 'max_output_tokens': 50}
        )
        print(f"   ✅ SUCCESS! Response:  {response.text. strip()}")
        working_models.append(model_name)
    except Exception as e:
        error_msg = str(e)
        if '404' in error_msg: 
            print(f"   ❌ Model not found")
        elif 'quota' in error_msg. lower():
            print(f"   ⚠️  Quota exceeded")
        elif 'permission' in error_msg.lower():
            print(f"   ⚠️  Permission denied")
        else:
            print(f"   ❌ Error: {error_msg[: 100]}")

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

if working_models:
    print(f"\n✅ WORKING VISION MODELS ({len(working_models)}):")
    for wm in working_models:
        print(f"  🎯 {wm}")
    
    print(f"\n💡 RECOMMENDED:  Use '{working_models[0]}' in your code")
else:
    print("\n❌ No working vision models found!")
    print("\nPossible issues:")
    print("  1. API key doesn't have access to vision models")
    print("  2. Region restrictions")
    print("  3. Need to enable Gemini API in Google Cloud Console")
    print("\nSteps to fix:")
    print("  1. Go to:  https://makersuite.google.com/app/apikey")
    print("  2. Create a new API key")
    print("  3. Enable 'Generative Language API'")
    print("  4. Try again")

print("\n" + "=" * 80)