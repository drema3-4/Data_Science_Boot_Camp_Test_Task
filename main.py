from services.store_service import StoreService


store_service = StoreService()

while True:
    user_command = input(">>> ").strip()

    if user_command == "help":
        print(
            "Загрузить документы: load_documents\n"
            "Вывести найденные документы: print\n"
            "Вывести только id найденных документов: print_only_ids"
        )
    elif user_command == "load_documents":
        data_path = input("Enter data_path\n>> ")

        store_service.load_data_in_store(data_path)
    elif user_command == "print":
        query = input("Enter query\n>> ")

        store_service.hybrid_article_search_print(query)
    elif user_command == "print_only_ids":
        query = input("Enter query\n>> ")

        store_service.hybrid_article_search_only_article_ids_print(query)
    elif user_command == "stop":
        break
    else:
        print("Такой кооманды нет")