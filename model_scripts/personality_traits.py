import pickle

def traits_predict(text, loaded_models):
    t_lake = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

    # print(y.columns)
    # for trait in t_lake:
    #     model_filename = f"../models/{trait}_model.pkl"
    #     with open(model_filename, 'rb') as file:
    #         loaded_models[trait] = pickle.load(file)

    # Example predictions using loaded models
    example_text = [text]

    result = []
    print("Example Predictions using Loaded Models:")
    for trait, model in loaded_models.items():
        example_pred = model.predict(example_text)
        result.append(example_pred[0])
        print(trait + ":", example_pred[0])

    inclination = t_lake[result.index(max(result))]
    return [inclination, result]