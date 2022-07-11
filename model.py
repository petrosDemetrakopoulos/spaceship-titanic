import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import feature_column
from tensorflow.python.keras import layers
from sklearn.model_selection import train_test_split

test_set = []
train_set = []
test_passenger_ids = []


def importDatasets():
    global test_set
    global train_set
    global test_passenger_ids
    test_set = pd.read_csv('data/test.csv')
    train_set = pd.read_csv('data/train.csv')
    test_passenger_ids = test_set['PassengerId']


def labelEncodeCategorical():
    global test_set
    global train_set
    # Label Encode the following features:
    # Homeplanet, Deck, Destination, Route, Side

    le = LabelEncoder()

    train_set['HomePlanet'] = le.fit_transform(train_set['HomePlanet'])
    train_set['Deck'] = le.fit_transform(train_set['Deck'])
    train_set['Destination'] = le.fit_transform(
        train_set['Destination'])
    train_set['Route'] = le.fit_transform(train_set['Route'])
    train_set['Side'] = le.fit_transform(train_set['Side'])

    test_set['HomePlanet'] = le.fit_transform(test_set['HomePlanet'])
    test_set['Deck'] = le.fit_transform(test_set['Deck'])
    test_set['Destination'] = le.fit_transform(test_set['Destination'])
    test_set['Route'] = le.fit_transform(test_set['Route'])
    test_set['Side'] = le.fit_transform(test_set['Side'])


def fillNaNumerical():
    global test_set
    global train_set

    train_set['Age'] = train_set['Age'].fillna(train_set['Age'].mean())
    train_set['RoomService'] = train_set['RoomService'].fillna(
        train_set['RoomService'].mean())
    train_set['FoodCourt'] = train_set['FoodCourt'].fillna(
        train_set['FoodCourt'].mean())
    train_set['ShoppingMall'] = train_set['ShoppingMall'].fillna(
        train_set['ShoppingMall'].mean())
    train_set['VRDeck'] = train_set['VRDeck'].fillna(
        train_set['VRDeck'].mean())
    train_set['Spa'] = train_set['Spa'].fillna(train_set['Spa'].mean())

    test_set['Age'] = test_set['Age'].fillna(test_set['Age'].mean())
    test_set['RoomService'] = test_set['RoomService'].fillna(
        test_set['RoomService'].mean())
    test_set['FoodCourt'] = test_set['FoodCourt'].fillna(
        test_set['FoodCourt'].mean())
    test_set['ShoppingMall'] = test_set['ShoppingMall'].fillna(
        test_set['ShoppingMall'].mean())
    test_set['VRDeck'] = test_set['VRDeck'].fillna(test_set['VRDeck'].mean())
    test_set['Spa'] = test_set['Spa'].fillna(test_set['Spa'].mean())


def fillNaCategorical():
    global test_set
    global train_set

    train_set = train_set.fillna({
        'HomePlanet': 'Earth',
        'CryoSleep': False,
        'Destination': 'TRAPPIST-1e',
        'VIP': False
    })

    test_set = test_set.fillna({
        'HomePlanet': 'Earth',
        'CryoSleep': False,
        'Destination': 'TRAPPIST-1e',
        'VIP': False
    })


def floatToInt():
    global test_set
    global train_set

    money_spent_columns = ['RoomService',
                           'FoodCourt', 'ShoppingMall', 'VRDeck', 'Spa']
    train_set[money_spent_columns] = train_set[money_spent_columns].astype(int)
    test_set[money_spent_columns] = test_set[money_spent_columns].astype(int)
    train_set['Age'] = train_set['Age'].astype(int)
    test_set['Age'] = test_set['Age'].astype(int)


def preprocess():
    global test_set
    global train_set
    global test_passenger_ids

    test_set = test_set.drop(['Name'], axis=1)
    train_set = train_set.drop(['Name'], axis=1)

    train_set[['PassengerGroup', 'PassengerNo']
              ] = train_set['PassengerId'].str.split('_', expand=True)
    test_set[['PassengerGroup', 'PassengerNo']
             ] = test_set['PassengerId'].str.split('_', expand=True)

    train_set['GroupSize'] = train_set.groupby(
        'PassengerGroup')['PassengerId'].transform('nunique')
    test_set['GroupSize'] = test_set.groupby(
        'PassengerGroup')['PassengerId'].transform('nunique')

    train_set[['Deck', 'Num', 'Side']] = train_set['Cabin'].str.split(
        '/', expand=True).fillna('Missing')
    test_set[['Deck', 'Num', 'Side']] = test_set['Cabin'].str.split(
        '/', expand=True).fillna('Missing')

    train_set['Route'] = train_set['HomePlanet'] + train_set['Destination']
    test_set['Route'] = test_set['HomePlanet'] + test_set['Destination']

    train_set = train_set.drop(
        ['Num', 'Cabin', 'PassengerNo', 'PassengerId', 'PassengerGroup'], axis=1)
    test_set = test_set.drop(
        ['Num', 'Cabin', 'PassengerNo', 'PassengerId', 'PassengerGroup'], axis=1)

    labelEncodeCategorical()

    # Floats to Int
    # replace NaN values
    fillNaNumerical()
    floatToInt()
    fillNaCategorical()

    train_set['Transported'] = train_set['Transported'].replace(
        {True: int(1), False: int(0)}).astype(int)
    train_set['VIP'] = train_set['VIP'].replace(
        {True: int(1), False: int(0)}).astype(int)

    test_set['VIP'] = test_set['VIP'].replace(
        {True: 1, False: 0}).astype(int)
    train_set['CryoSleep'] = train_set['CryoSleep'].replace(
        {True: int(1), False: int(0)}).astype(int)
    test_set['CryoSleep'] = test_set['CryoSleep'].replace(
        {True: int(1), False: int(0)}).astype(int)

    train_set['isOverNine'] = 1
    train_set.loc[train_set['Age'] <= 10, 'isOverNine'] = 0

    test_set['isOverNine'] = 1
    test_set.loc[train_set['Age'] <= 10, 'isOverNine'] = 0

    train_set['TotalSpending'] = train_set['Spa'] + train_set['ShoppingMall'] + \
        train_set['RoomService'] + train_set['FoodCourt'] + train_set['VRDeck']
    test_set['TotalSpending'] = test_set['Spa'] + test_set['ShoppingMall'] + \
        test_set['RoomService'] + test_set['FoodCourt'] + test_set['VRDeck']


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop('Transported')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def extractFeatureColumns(train):
    feature_columns = []
    numeric_cols = ['Age', 'RoomService',
                    'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending', 'GroupSize']
    categorical_cols = ['HomePlanet', 'CryoSleep',
                        'Destination', 'VIP', 'Route', 'Deck', 'Side', 'isOverNine']
    for feature in numeric_cols:
        feature_columns.append(feature_column.numeric_column(feature))

    for feature in categorical_cols:
        categorical_column = feature_column.categorical_column_with_vocabulary_list(
            feature, train[feature].unique())
        indicator_column = feature_column.indicator_column(categorical_column)
        feature_columns.append(indicator_column)
    return feature_columns


def __main__():
    importDatasets()
    preprocess()

    # train / test split
    train, test = train_test_split(train_set, test_size=0.1)

    print(len(train), 'train examples')
    print(len(test), 'test examples')

    feature_columns = extractFeatureColumns(train)
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    test_set['Transported'] = 0
    val_ds = df_to_dataset(test_set, shuffle=False, batch_size=len(test_set))

    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(.1),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=test_ds,
              epochs=50)

    loss, accuracy = model.evaluate(test_ds)
    print('')
    print("Accuracy", accuracy)
    print('')

    test_set['Pred'] = model.predict(val_ds).ravel()
    test_set['Transported'] = 'True'
    test_set.loc[test_set['Pred'] <= 0.5, 'Transported'] = 'False'

    print(test_set)
    repsonse_frame = pd.DataFrame(
        data={'PassengerId': test_passenger_ids, 'Transported': test_set['Transported']})
    repsonse_frame.to_csv('data/submission.csv', index=False)


__main__()
