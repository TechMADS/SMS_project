import os
from huggingface_hub import InferenceClient
from PIL import Image
import numpy as np
import io

# Your token is correctly included
HF_TOKEN = "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"

print("🧪 Testing Hugging Face Client with 'Inference Providers'...")

try:
    # Initialize the client
    client = InferenceClient(
        provider="hf-inference",
        token=HF_TOKEN
    )
    print("✅ Client initialized successfully.")

    print("\n1. Testing Image Classification...")
    # Create a simple 224x224 random color image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Convert PIL Image to bytes (the format the API expects)
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format='JPEG')
    image_bytes = img_byte_arr.getvalue()
    
    # Perform image classification with bytes
    results = client.image_classification(
        image=image_bytes,  # Pass bytes instead of PIL Image
        model="google/vit-base-patch16-224"
    )
    print(f"✅ Image classification successful!")
    for result in results[:3]:  # Show top 3 predictions
        print(f"   Label: {result.label}, Score: {result.score:.4f}")

    print("\n2. Testing with a simpler model...")
    # Let's also test with a smaller, faster model
    results2 = client.image_classification(
        image=image_bytes,
        model="microsoft/resnet-50"
    )
    print(f"✅ ResNet-50 classification successful!")
    for result in results2[:2]:
        print(f"   Label: {result.label}, Score: {result.score:.4f}")

    print("\n3. Testing Zero-Shot Classification (useful for dress code)...")
    # This is actually more relevant for your project!
    candidate_labels = ["formal attire", "casual clothing", "person with tie", "neat appearance"]
    zs_results = client.zero_shot_image_classification(
        image=image_bytes,
        candidate_labels=candidate_labels,
        model="openai/clip-vit-large-patch14-336"
    )
    print(f"✅ Zero-shot classification successful!")
    for result in zs_results:
        print(f"   '{result.label}': {result.score:.4f}")

    print("\n🎉 All tests passed! Your Hugging Face API integration is working.")
    print("   You can now integrate this into your student monitoring system.")

except Exception as e:
    print(f"\n❌ A test failed with error: {type(e).__name__}")
    print(f"   Details: {e}")
    
    # More specific error handling
    if "401" in str(e) or "403" in str(e):
        print("\n💡 Authentication error. Check your token permissions.")
        print("   Ensure it has 'Inference API' permission at: https://huggingface.co/settings/tokens")
    elif "timeout" in str(e).lower():
        print("\n💡 Timeout error. The model might be loading. Try again in 30 seconds.")
    elif "model" in str(e).lower() and "not found" in str(e).lower():
        print("\n💡 Model not found. Try a different model name.")