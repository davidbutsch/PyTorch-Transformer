from generator import Generator


def main():
    generator = Generator()

    while True:
        prompt = input("")

        response = generator.generate(prompt)

        print(response)


if __name__ == "__main__":
    main()
