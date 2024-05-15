import os

menu_options = {
    1: 'Modelo TechGenius',
    2: 'Modelo Transfer learning',
    0: 'Salir'
}

def print_menu():
    for key in menu_options.keys():
        print(key, ': ', menu_options[key])

print_menu()
option = int(input('Escoga un modelo con el que conversar: '))

while option != 0:
    if option == 1:
        # Modelo TechGenius
        print("Seleccionaste el Modelo TechGenius")
    elif option == 2:
        # Modelo Transfer learning
        print("Seleccionaste el Modelo Transfer learning")
    elif option == 0:
        # Salir
        print("Saliendo del programa...")
    else:
        os.system('cls')

# Add a prompt to keep the terminal window open
input("Presiona Enter para salir...")

