import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


def plot_results(prediction, O_Error):
    X_label=['GBR', 'SVM', 'LR', 'RFR']
    barWidth = 0.2
    bar_num=len(O_Error)
    fig = plt.figure(figsize=(15, 10))
    # set height of bar
    bars=[]
    bar1 = np.arange(len(O_Error[0]))
    bars.append(bar1)
    for i in np.arange(bar_num):
        bar = [x + (i+1)*barWidth for x in bar1]
        bars.append(bar)

    # Make the plot
    plt.rcParams.update({'font.size': 30})
    hatchs=['/', 'o', '*', '/']
    for i in np.arange(bar_num):
        plt.bar(bars[i], O_Error[i] , width=barWidth,
            edgecolor='grey', label='FG%s' %(i), hatch=hatchs[i])

    # Adding Xticks
    #plt.xlabel(X_label)
    plt.xticks(bars[0],X_label)
    plt.legend(fontsize=25, loc='lower left')
    plt.savefig("RMSE of %s" %(prediction))
    plt.close()

def feature_group(dataset_name):
    if dataset_name == "origin":
        FG_list = []
        FG1 = ['PDOY', 'rMG', 'ED_R1_Tmax', 'R1_R5_Tmax', 'R5_R7_Tmax', 'ED_R1_Tmin', 'R1_R5_Tmin', 'R5_R7_Tmin',
               'ED_R1_Srad', 'R1_R5_Srad', 'R5_R7_Srad']
        #  MAX_Tmax_ED_R1, MIN_Tmax_ED_R1, SD_Tmax_ED_R1, MAX_Tmin_ED_R1, MIN_Tmin_ED_R1, SD_Tmin_ED_R1, MAX_Srad_ED_R1, MIN_Srad_ED_R1, SD_Srad_ED_R1, MAX_Phot_ED_R1, MIN_Phot_ED_R1, SD_Phot_ED_R1, MAX_Tmax_R1_R5, MIN_Tmax_R1_R5, SD_Tmax_R1_R5, MAX_Tmin_R1_R5, MIN_Tmin_R1_R5, SD_Tmin_R1_R5, MAX_Srad_R1_R5, MIN_Srad_R1_R5, SD_Srad_R1_R5, MAX_Phot_R1_R5, MIN_Phot_R1_R5, SD_Phot_R1_R5, MAX_Tmax_R5_R7, MIN_Tmax_R5_R7, SD_Tmax_R5_R7, MAX_Tmin_R5_R7, MIN_Tmin_R5_R7, SD_Tmin_R5_R7, MAX_Srad_R5_R7, MIN_Srad_R5_R7, SD_Srad_R5_R7, MAX_Phot_R5_R7, MIN_Phot_R5_R7, SD_Phot_R5_R7
        sim_model_features = ['Dens', 'ED_R1_Phot', 'R1_R5_Phot', 'R5_R7_Phot', 'CIPAR_E_R1', 'CIPAR_R1_R5',
                              'CIPAR_R5_R7']
        FG2 = FG1 + sim_model_features
        FG3 = FG2 + ['ED', 'R1', 'R5', 'R7']
        FG4 = FG3 + ['MAX_Tmax_ED_R1', 'MIN_Tmax_ED_R1', 'SD_Tmax_ED_R1', 'MAX_Tmin_ED_R1', 'MIN_Tmin_ED_R1',
                     'SD_Tmin_ED_R1', 'MAX_Srad_ED_R1', 'MIN_Srad_ED_R1', 'SD_Srad_ED_R1', 'MAX_Phot_ED_R1',
                     'MIN_Phot_ED_R1', 'SD_Phot_ED_R1', 'MAX_Tmax_R1_R5', 'MIN_Tmax_R1_R5', 'SD_Tmax_R1_R5',
                     'MAX_Tmin_R1_R5', 'MIN_Tmin_R1_R5', 'SD_Tmin_R1_R5', 'MAX_Srad_R1_R5', 'MIN_Srad_R1_R5',
                     'SD_Srad_R1_R5', 'MAX_Phot_R1_R5', 'MIN_Phot_R1_R5', 'SD_Phot_R1_R5', 'MAX_Tmax_R5_R7',
                     'MIN_Tmax_R5_R7', 'SD_Tmax_R5_R7', 'MAX_Tmin_R5_R7', 'MIN_Tmin_R5_R7', 'SD_Tmin_R5_R7',
                     'MAX_Srad_R5_R7', 'MIN_Srad_R5_R7', 'SD_Srad_R5_R7', 'MAX_Phot_R5_R7', 'MIN_Phot_R5_R7',
                     'SD_Phot_R5_R7']
        FG_list.append(FG1)
        FG_list.append(FG2)
        FG_list.append(FG3)
        FG_list.append(FG4)
    elif dataset_name =="aggre":
        FG_list = []
        FG1 = ['PDOY', 'rMG', 'ED_R1_Tmax', 'R1_R5_Tmax', 'R5_R7_Tmax', 'ED_R1_Tmin', 'R1_R5_Tmin', 'R5_R7_Tmin',
               'ED_R1_Srad', 'R1_R5_Srad', 'R5_R7_Srad']
        #  MAX_Tmax_ED_R1, MIN_Tmax_ED_R1, SD_Tmax_ED_R1, MAX_Tmin_ED_R1, MIN_Tmin_ED_R1, SD_Tmin_ED_R1, MAX_Srad_ED_R1, MIN_Srad_ED_R1, SD_Srad_ED_R1, MAX_Phot_ED_R1, MIN_Phot_ED_R1, SD_Phot_ED_R1, MAX_Tmax_R1_R5, MIN_Tmax_R1_R5, SD_Tmax_R1_R5, MAX_Tmin_R1_R5, MIN_Tmin_R1_R5, SD_Tmin_R1_R5, MAX_Srad_R1_R5, MIN_Srad_R1_R5, SD_Srad_R1_R5, MAX_Phot_R1_R5, MIN_Phot_R1_R5, SD_Phot_R1_R5, MAX_Tmax_R5_R7, MIN_Tmax_R5_R7, SD_Tmax_R5_R7, MAX_Tmin_R5_R7, MIN_Tmin_R5_R7, SD_Tmin_R5_R7, MAX_Srad_R5_R7, MIN_Srad_R5_R7, SD_Srad_R5_R7, MAX_Phot_R5_R7, MIN_Phot_R5_R7, SD_Phot_R5_R7
        sim_model_features = ['Dens', 'ED_R1_Phot', 'R1_R5_Phot', 'R5_R7_Phot', 'CIPAR_E_R1', 'CIPAR_R1_R5',
                              'CIPAR_R5_R7']
        FG2 = FG1 + sim_model_features
        FG3 = FG2 + ['Biomass_SIM', 'LAI_SIM', 'Yield_SIM', 'Oil_SIM', 'Protein_SIM']
        # FG4=FG3+['MAX_Tmax_ED_R1', 'MIN_Tmax_ED_R1', 'SD_Tmax_ED_R1', 'MAX_Tmin_ED_R1', 'MIN_Tmin_ED_R1', 'SD_Tmin_ED_R1', 'MAX_Srad_ED_R1', 'MIN_Srad_ED_R1', 'SD_Srad_ED_R1', 'MAX_Phot_ED_R1', 'MIN_Phot_ED_R1', 'SD_Phot_ED_R1', 'MAX_Tmax_R1_R5', 'MIN_Tmax_R1_R5', 'SD_Tmax_R1_R5', 'MAX_Tmin_R1_R5', 'MIN_Tmin_R1_R5', 'SD_Tmin_R1_R5', 'MAX_Srad_R1_R5', 'MIN_Srad_R1_R5', 'SD_Srad_R1_R5', 'MAX_Phot_R1_R5', 'MIN_Phot_R1_R5', 'SD_Phot_R1_R5', 'MAX_Tmax_R5_R7', 'MIN_Tmax_R5_R7', 'SD_Tmax_R5_R7', 'MAX_Tmin_R5_R7', 'MIN_Tmin_R5_R7', 'SD_Tmin_R5_R7', 'MAX_Srad_R5_R7', 'MIN_Srad_R5_R7', 'SD_Srad_R5_R7', 'MAX_Phot_R5_R7', 'MIN_Phot_R5_R7', 'SD_Phot_R5_R7']
        FG_list.append(FG1)
        FG_list.append(FG2)
        FG_list.append(FG3)
    return FG_list


def model_selection(models, o_dict, X):
    overall_results=[]
    for model in models:
        print(model)
        results=[]
        for key in o_dict.keys():
            steps = [('scaler', StandardScaler()), ('model', model)]
            pipeline = Pipeline(steps)
            wrapped_model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
            scores = cross_val_score(wrapped_model, X, o_dict[key], cv=5, scoring="neg_root_mean_squared_error")
            scores=abs(scores)
            print("Cross Validation RMSE Score for %s:" %(key))
            print(scores)
            averaged_score=sum(scores)/len(scores)
            print("Averaged RMSE: %f" %(averaged_score))
            results.append(averaged_score)
        overall_results.append(results)
        print('--------------------------------------------------------------')
    return overall_results


def main():
    dataset_name="origin"
    if dataset_name=="origin":
        soybean_df = pd.read_csv('ML_Data_09162022.csv')
        soybean_df.drop(soybean_df[soybean_df["rMG1"] == 7].index, inplace=True)
        outputs = ['Yield', 'Oil', 'Protein']
    elif dataset_name=="aggre":
        soybean_df = pd.read_excel('ML_DB_V1.xlsx')
        outputs = ['Yield_OBS', 'Oil_OBS', 'Protein_OBS']

    soybean_df.dropna(inplace=True)
    FG_list = feature_group(dataset_name)
    Yields_Error = []
    Oil_Error = []
    Protein_Error = []
    for fg in FG_list:
        X_features = fg
        print(X_features)
        X = soybean_df[X_features]
        Y_Yield = soybean_df[outputs[0]]
        Y_Oil = soybean_df[outputs[1]]
        Y_Protein = soybean_df[outputs[2]]
        o_dict = {'o_yields': Y_Yield, 'o_Oil': Y_Oil, 'o_Protein': Y_Protein}
        models = [GradientBoostingRegressor(), LinearSVR(), LinearRegression(), RandomForestRegressor()]
        final_results = model_selection(models, o_dict, X)
        print(final_results)
        transposed_final_results = np.array(final_results).T
        print(transposed_final_results)
        Yields_Error.append(transposed_final_results[0])
        Oil_Error.append(transposed_final_results[1])
        Protein_Error.append(transposed_final_results[2])
    plot_results("Yield", Yields_Error)
    plot_results("Oil", Oil_Error)
    plot_results("Protein", Protein_Error)

if __name__ == "__main__":
    main()


