def get_data():
    import os
    import pandas as pd
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    file_path = os.path.join(data_folder, 'data_2.xlsx')

    df = pd.read_excel(file_path)
    df.dropna(subset=['Sentiment'], inplace=True)
    df = df.iloc[:70]
    labels = df['Sentiment'].tolist()
    labels = [int(round((x+1)/2*4)) for x in labels]
    return df['Review Text'].tolist(), labels
    




if __name__ == "__main__":
    get_data()