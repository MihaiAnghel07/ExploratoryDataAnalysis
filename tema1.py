import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import scipy.stats as stats
import seaborn as sns


def read_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def replace_nan_with(current_el, new_el):
    if current_el is np.nan:
        return new_el
    else:
        return current_el
    

def transform_data(json_data):
    # a. Din format JSON în formatul necesar pentru restul pipeline-ului
    df = pd.DataFrame(json_data)

    # b. Descoperirea și corectarea erorilor care au apărut din procedura de colectare
    print(df.shape)
    print(df.dtypes)
    print(df.dtypes.value_counts())

    # urmatoarea bucata de cod o voi tine comentata pentru ca dureaza foarte mult rularea, am folosit-o
    # pentru identificarea erorilor, dar nu este critica pentru obtinerea rezultatelor de analiza
     
    # for col in df.select_dtypes('object').columns:
    #     if not isinstance(df[col].iloc[0], list):
    #         print(col, df[col].unique())

    #         for el in df[col]:
    #             if isinstance(el, list):
    #                 print(col, set(el))
    #             else:
    #                 print(el)
    #     else:
    #         print(col, df[col].unique())

    #     print(col, df[col].unique())

    # for lista in df["Audio si tehnologie"]:
    #     if isinstance(lista, list):  
    #         print(set(lista)) 
    #     else:
    #         print(lista)

    df.rename(columns={'Anul fabricaÈ›iei': 'Anul fabricatiei'}, inplace=True)
    df['Marca'] = df['Marca'].replace('CitroÃ«n', 'Citroen')
    df['Valoare rata lunara'] = df['Valoare rata lunara'].str.extract('(\d+)').astype(float)
    df['Plata initiala (la predare)'] = df['Plata initiala (la predare)'].str.extract('(\d+)').astype(float)
    df['Valoare reziduala'] = df['Valoare reziduala'].str.extract('(\d+)').astype(float)
    df['Consum Mixt'] = df['Consum Mixt'].str.extract('(\d+)').astype(float)
    df['Garantie dealer (inclusa in pret)'] = df['Garantie dealer (inclusa in pret)'].str.extract('(\d+)').astype(float)


    # singurul field care ori are o lista de elemente, ori e NaN, motiv pentru care trebuie pus separat, altfel primesc exceptie
    df["Performanta"] = df["Performanta"].apply(replace_nan_with, args=(["indisponibil"],))

    # iterez prin variabilele te dip object si inlocuiesc valorile NaN cu "indisponibil" sau ["indisponibil"], dupa caz
    for col in df.select_dtypes('object').columns:
        if isinstance(df[col], list) or isinstance(df[col].iloc[0], list):
            df[col] = df[col].apply(replace_nan_with, args=(["indisponibil"],))
        else:
            df[col] = df[col].apply(replace_nan_with, args=("indisponibil",))            


    # iterez prin variabilele te dip float64 si int64 si inlocuiesc valorile NaN cu media valorilor pe coloana respectiva
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df[col].fillna(df[col].mean(), inplace=True)
    

    # este mai logic ca aceasta coloana sa fie de tipul int
    df['Numar locuri'] = df['Numar locuri'].astype(int)


    # scalez valorile coloanelor numerice pentru a elimina valorile outliers
    # in functie de fiecare categorie in parte, setez anumite limite logice pentru ca 
    # altfel IQR poate sa puna anumite valori gresite 
    df = normalize_column(df, "pret", 0, 250000)
    df = normalize_column(df, "Anul fabricatiei", 1970, 2024)
    df = normalize_column(df, "Km", 0, 400000)
    df = normalize_column(df, "Putere", 0, 500)
    df = normalize_column(df, "Capacitate cilindrica", 990, 5000)
    df = normalize_column(df, "Consum Extraurban", 2, 15)
    df = normalize_column(df, "Consum Urban", 2, 15)
    df = normalize_column(df, "Consum Mixt", 2, 15)
    df = normalize_column(df, "Emisii CO2", 0, 300)
    df = normalize_column(df, "Numar de portiere", 1, 6)
    df = normalize_column(df, "Numar locuri", 1, 8)
    df = normalize_column(df, "sau in limita a", 1, 300000)
    df = normalize_column(df, "Garantie dealer (inclusa in pret)", 0, 120)
    df = normalize_column(df, "Numar de rate lunare ramase", 1, 60)
    

    # c. Adăugarea sau eliminarea de coloane (acolo unde este cazul, de exemplu prin transformarea celor existente)
    #elimin coloanele cu un numar foarte mic de intrari sau nerelevante pentru analiza noastra
    df.drop("Timp de incarcare", axis=1, inplace=True)
    df.drop("Tuning", axis=1, inplace=True)
    df.drop("Contract baterie", axis=1, inplace=True)
    df.drop("Masina de epoca", axis=1, inplace=True)
    df.drop("Autonomie", axis=1, inplace=True)
    df.drop("Numar de rate lunare ramase", axis=1, inplace=True)
    df.drop("Valoare rata lunara", axis=1, inplace=True)
    df.drop("Plata initiala (la predare)", axis=1, inplace=True)
    df.drop("VIN (serie sasiu)", axis=1, inplace=True) # nu este nicio serie, doar un label 

    print(df.dtypes)

    return df


def normalize_column(df, column, min_value, max_value):
    Q1 = df[column].quantile(0.20)
    Q3 = df[column].quantile(0.80)

    IQR = Q3 - Q1

    lower_bound = max(Q1 - 1.5 * IQR, min_value)
    upper_bound = min(Q3 + 1.5 * IQR, max_value)
    
    # print(lower_bound)
    # print(upper_bound)

    # outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    # print(outliers.shape)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    


def main():
    
    # 1. Citirea și încărcarea datelor din fișierul la dispoziție
    json_data = read_data('tema-1-dmda-24.json')

    # 2. Transformarea datelor
    df = transform_data(json_data)
    
    # 3. Analiza datelor

    # analizam cateva coloane cu date categorice, folosind bar chart
    # am impartit cele 4 coloane in 1 + 3 deoarece coloana "Marca" contine foarte multe tipuri de marci si
    # daca as fi grupat toate graficele, nu s-ar fi inteles nimic din graficul acesta

    cars_brands = df["Marca"].value_counts()
    cars_brands.plot(kind='bar', color='blue', edgecolor='black', linewidth=0.5, width=0.5)
    plt.grid(alpha=0.75)
    plt.title('Number of cars per brand')
    plt.xlabel('Brand')
    plt.ylabel('Count')
    plt.show()

    cols = [ "Combustibil", "Transmisie", "Norma de poluare"]
    fig, ax = plt.subplots(3, 1, figsize=(8, 16), dpi=100)
    plt.rcParams['font.size'] = 6
    for idx, col in enumerate(cols):
        filtered_df = df[df[col] != "indisponibil"]
        counts = filtered_df[col].value_counts()
        ax[idx].bar(counts.index, counts.values, alpha=0.75, label=f'{col} counts', color='blue', edgecolor='black', linewidth=0.5, width=0.5)
        ax[idx].set_ylabel('Count')
        ax[idx].grid(alpha=0.75)
        ax[idx].legend()
    plt.tight_layout()
    plt.show()


    # facem acelasi lucru, dar pentru date numerice, folosind ball chart

    cols = ["pret", "Putere", "Anul fabricatiei"]
    fig, ax = plt.subplots(3, 1, figsize=(8, 16), dpi=100)
    plt.rcParams['font.size'] = 6
    for idx, col in enumerate(cols):
        counts = df[col].value_counts()
        ax[idx].scatter(counts.index, np.zeros(len(counts)), alpha=0.75, label=f'{col} counts', color='blue', edgecolor='black', linewidth=0.5, s=counts.values)
        ax[idx].set_ylabel('Count Radius (scaled)')
        ax[idx].set_xlabel(col)
        ax[idx].grid(alpha=0.75)
        ax[idx].legend()
    plt.tight_layout()
    plt.show()


    # trasez diagrama cumulativa pentru coloana "pret", este unul dintre cei mai importanti factori pentru vanzarea unei masini

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
    plt.hist(df['pret'], bins=100, cumulative=True, density=True, alpha=0.65, label='Cumulative Histogram', color='blue', edgecolor='black', linewidth=0.5, rwidth=0.85)
    plt.title('Cumulative Histogram of Cars price')
    plt.ylabel('Cumulative Density')
    plt.xlabel('Price')
    plt.grid(alpha=0.75)
    plt.legend()
    plt.show()


    # generez o duagrama bar chart pentru pret < 40.000 euro deoarece acesta este segmentul de pret cel mai comun al masinior de vanzare
    # detaliem un pic diagrama adaugand linia de distributie normala, pretul mediu si deviatia standard pentru pretul mediu

    truncated_df = df[df['pret'] < 40000]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
    plt.rcParams['font.family'] = 'monospace'
    plt.hist(truncated_df['pret'], bins=50, alpha=0.65, color='blue', edgecolor='black', linewidth=0.5, label='Price', rwidth=0.9, density=True)
    plt.axvline(truncated_df['pret'].mean(), color='red', linestyle='--', label='Mean price')
    x = np.linspace(truncated_df['pret'].min(), truncated_df['pret'].max(), 100)
    y = stats.norm.pdf(x, truncated_df['pret'].mean(), truncated_df['pret'].std())
    plt.plot(x, y, color='red', linestyle='-', label='Normal Distribution')
    plt.axvline(truncated_df['pret'].mean() + truncated_df['pret'].std(), color='red', linestyle='dotted', label='Mean price +/-1 std')
    plt.axvline(truncated_df['pret'].mean() - truncated_df['pret'].std(), color='red', linestyle='dotted')
    plt.title('Price Histogram (price < 40000)')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.grid(alpha=0.75)
    plt.legend()
    plt.show()


    # graficul urmator cuprinde cateva coloane cu date de tip object: Transmisie, Stare, Tip Caroserie, combustibil
    # se pot trasa graficele in grupuri de cate 2 pentru a fi mai lizibile
    # graficul ilustreaza fiecare categorie ca doua bare de culori diferite, una pentru inregistrari cu pretul
    # sub 20.000, cealalta pentru pret >= 20.000 
    
    plt.rcParams['font.family'] = 'monospace'
    low_prices = df[df['pret'] < 20000 ]
    high_prices = df[df['pret'] >= 20000]
    categories = ["Transmisie", "Stare", "Tip Caroserie", "Combustibil"]
    fig, ax = plt.subplots(1, len(categories), figsize=(20, 5), dpi=100)
    for i, category in enumerate(categories):
        low_counts = low_prices[category].value_counts()
        high_counts = high_prices[category].value_counts()
        low_counts_dict = low_counts.to_dict()
        high_counts_dict = high_counts.to_dict()
        for key in low_counts_dict.keys():
            if key not in high_counts_dict:
                high_counts_dict[key] = 0
        for key in high_counts_dict.keys():
            if key not in low_counts_dict:
                low_counts_dict[key] = 0
        # sort the values by the key
        low_counts_dict = dict(sorted(low_counts_dict.items()))
        high_counts_dict = dict(sorted(high_counts_dict.items()))
        # get the values of the counts
        low_counts_values = list(low_counts_dict.values())
        high_counts_values = list(high_counts_dict.values())
        # get the keys of the counts
        low_counts_keys = list(low_counts_dict.keys())
        high_counts_keys = list(high_counts_dict.keys())
        ind = np.arange(len(low_counts_keys))
        width = 0.35
        ax[i].bar(ind, low_counts_values, width, alpha=0.65, color='blue', edgecolor='black', linewidth=0.5, label='Price < 20000')
        ax[i].bar(ind + width, high_counts_values, width, alpha=0.65, color='red', edgecolor='black', linewidth=0.5, label='Price >= 20000')
        ax[i].set_xticks(ind + width / 2)
        ax[i].set_xticklabels(low_counts_keys)
        ax[i].set_title(category)
        ax[i].set_xlabel('Value')
        ax[i].set_ylabel('Count')
        ax[i].grid(alpha=0.75)
        ax[i].legend()
    plt.tight_layout()
    plt.show()


    # generam si cateva grafice de dispersie pentru cateva coloane de tip numeric: Consum Urban, Capacitate cilindrica, Anul fabricatiei, Km
    # acest tip de grafic ne ajuta sa observam daca exista relatie de liniaritate intre variabilele alese si pret 
    # inregistrarile cu pret sub 20.000 sunt colorate albastru, celelalte cu rosu
 
    plt.rcParams['font.family'] = 'monospace'
    low_prices = df[df['pret'] < 20000]
    high_prices = df[df['pret'] >= 20000]
    numerical_categories = ['Consum Urban', 'Capacitate cilindrica', 'Anul fabricatiei', 'Km']
    fig, ax = plt.subplots(1, len(numerical_categories), figsize=(20, 5), dpi=100)
    for i, variable in enumerate(numerical_categories):
        ax[i].scatter(low_prices[variable], low_prices['pret'], alpha=0.65, color='blue', edgecolor='black', linewidth=0.5, label='Price < 20000')
        ax[i].scatter(high_prices[variable], high_prices['pret'], alpha=0.65, color='red', edgecolor='black', linewidth=0.5, label='Price >= 20000')
        ax[i].set_title(variable)
        ax[i].set_xlabel(variable)
        ax[i].set_ylabel('Price')
        ax[i].grid(alpha=0.75)
    plt.show()


    # pentru a observa daca exista corelatie intre variabilele numerice, trasam doua grafice: Pearson si Spearman
    # pastram doar corelatiile relevante de peste 0.4

    fig, ax = plt.subplots(1, 2, figsize=(30, 10), dpi=100)
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.size'] = 6
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_df = df[numeric_columns]
    for i, corr_type in enumerate(['pearson', 'spearman']):
        corr_df = numeric_df.corr(corr_type)
        corr_df = corr_df - np.diag(np.diag(corr_df))
        corr_df = corr_df[corr_df > 0.4]
        corr_df = corr_df.dropna(axis=0, how='all')
        corr_df = corr_df.dropna(axis=1, how='all')
        sns.heatmap(corr_df, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, linecolor='black', square=False, cbar=True, cbar_kws={'orientation': 'vertical', 'shrink': 0.8, 'pad': 0.05}, ax=ax[i], mask=corr_df.isnull())
        ax[i].set_title(corr_type, fontsize=20)
    plt.tight_layout()
    plt.show()

    print(df.describe())
        


if __name__ == "__main__":
    main()