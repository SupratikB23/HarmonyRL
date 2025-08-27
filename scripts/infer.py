from harmonyrl.inference import generate

if __name__ == "__main__":
    # toggle use_diffusers=True once you have the weights/GPU ready
    out = generate(use_diffusers=False)
    print("Saved:", out)