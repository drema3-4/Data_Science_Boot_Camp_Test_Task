from services.models_manager import ModelsManager
from services.store_manager import StoreManager
from services.store_service import StoreService
from services.experiment_service import ExperimentService


models_manager = ModelsManager()
store_manager = StoreManager(models_manager)
store_service = StoreService(store_manager, models_manager)
experiment_service = ExperimentService(store_service)

while True:
    user_command = input(">>> ").strip()

    if user_command == "help":
        print(
            "Загрузить документы: load_documents\n"
            "Вывести найденные документы: print\n"
            "Вывести только id найденных документов: print_only_ids\n"
            "Рассчитать mean recall@k: mean_recall_k\n"
            "Рассчитать mean AP@k: mean_avg_precision_k\n"
            "Рассчитать метрики для модели: test_model\n"
            "Остановить цикл выполнения: stop"
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
    elif user_command == "mean_recall_k":
        calibration_data_path = input("Enter calibration_data_path\n>> ")
        k = int(input("Enter k\n>> "))

        print(experiment_service.mean_recall_k(calibration_data_path, k))
    elif user_command == "mean_avg_precision_k":
        calibration_data_path = input("Enter calibration_data_path\n>> ")
        k = int(input("Enter k\n>> "))

        print(experiment_service.mean_average_precision_k(calibration_data_path, k))
    elif user_command == "test_model":
        calibration_data_path = input("Enter calibration_data_path\n>> ")
        
        k = int(input("Enter k\n>> "))
        print(f"recall@{k} = {experiment_service.mean_recall_k(calibration_data_path, k)}")

        k = int(input("Enter k\n>> "))
        print(f"MAP@{k} = {experiment_service.mean_average_precision_k(calibration_data_path, k)}")
    elif user_command == "stop":
        break
    else:
        print("Такой кооманды нет")