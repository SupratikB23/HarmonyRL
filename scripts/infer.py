from harmonyrl.inference import generate

if __name__ == "__main__":
    out = generate(use_diffusers=False)
    print("Saved:", out)