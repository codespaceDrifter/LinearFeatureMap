from phi4_mini.inference import Phi4Inference

def main():
    phi = Phi4Inference()
    print("Model loaded. Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        responses, _, _ = phi.generate([user_input], max_new_tokens=256)
        print(f"Phi4: {responses[0]}\n")

if __name__ == "__main__":
    main()