import argparse
import sys

def parse_type():
    parser = argparse.ArgumentParser(description="Ersten Typ abfragen.")
    parser.add_argument('--type', required=True, choices=['type1', 'type2'], help="Geben Sie den Typ an: 'type1' oder 'type2'.")
    
    # Parse nur die Argumente für diesen Parser
    args, remaining_args = parser.parse_known_args()
    
    # Zurückgeben des Typs und der verbleibenden Argumente
    return args.type, remaining_args

def parse_arguments(type_selected, remaining_args):
    parser = argparse.ArgumentParser(description="Weitere Argumente abfragen, abhängig vom Typ.")

    if type_selected == 'type1':
        parser.add_argument('--param1', type=int, default=10, help="Parameter 1 für Typ 1 (Standard: 10).")
        parser.add_argument('--param2', type=str, default='foo', help="Parameter 2 für Typ 1 (Standard: 'foo').")
    elif type_selected == 'type2':
        parser.add_argument('--param1', type=int, default=20, help="Parameter 1 für Typ 2 (Standard: 20).")
        parser.add_argument('--param2', type=str, default='bar', help="Parameter 2 für Typ 2 (Standard: 'bar').")

    # Parse die verbleibenden Argumente mit diesem Parser
    args = parser.parse_args(remaining_args)
    return args

def main():
    # Zuerst den Typ abfragen
    type_selected, remaining_args = parse_type()

    # Dann basierend auf dem Typ die weiteren Argumente abfragen
    args = parse_arguments(type_selected, remaining_args)

    # Skriptlogik hier
    print(f"Gewählter Typ: {type_selected}")
    print(f"Parameter 1: {args.param1}")
    print(f"Parameter 2: {args.param2}")

if __name__ == "__main__":
    main()
