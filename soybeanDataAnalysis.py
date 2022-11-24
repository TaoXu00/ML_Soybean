import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_locations_of_each_year(df, year):
    df_location=df.loc[(df['YEAR']==year), ['YEAR', 'Location']].drop_duplicates()
    return df_location

def find_PD_of_each_year_Location(df, year, location):
    df_PD=df.loc[(df['YEAR']==year) & (df['Location']==location), ['YEAR', 'Location', 'PD']].drop_duplicates()
    return df_PD
def find_Variety_of_each_year_PD(df, year,location, PD):
    df_Variety=df.loc[(df['YEAR']==year) &(df['PD']==PD) &(df['Location']==location),['YEAR', 'Location', 'PD', 'Variety']].drop_duplicates()
    return df_Variety

def find_Variety_of_each_location_all_year_all_period(df, location):
    for year in [2012, 2013, 2014]:
        for PD in [1, 2, 3, 4]:
            df_Variety = df.loc[(df['YEAR'] == year) & (df['PD'] == PD) & (df['Location'] == location), ['Variety']].drop_duplicates()
            print(df_Variety.values.tolist())
def find_Variety_of_differrent_location_in_the_same_year_in_the_same_period(df):
    for year in [2012, 2013, 2014]:
        print(year)
        for location in ['C St', 'Colu', 'Keis', 'Mila', 'Port', 'Rohw', 'St J', 'Ston', 'Vero']:
            df_Variety = df.loc[(df['YEAR'] == year) & (df['Location'] == location), ['Variety']].drop_duplicates()
            print(df_Variety['Variety'].values.tolist())

def find_Variety_of_same_location_same_period_but_differnt_year(df, location, PD):
    for year in [2012, 2013, 2014]:
        df_Variety = df.loc[(df['YEAR'] == year) & (df['Location'] == location) & (df['PD'] == PD), ['Variety']].drop_duplicates()
        print(df_Variety['Variety'].values.tolist())

def find_tempreture_in_differnt_year_differntlocation_differnt_period(df,location, year, Tempreture):
    labels=['PD=1', 'PD=2', 'PD=3', 'Pd=4']
    T_list=[]
    for PD in [1, 2, 3, 4]:
        dataframe=df.loc[(df['Location']==location) & (df['YEAR'] == year) & (df['PD'] == PD), ['PD', Tempreture]].drop_duplicates()
        T_list.append(dataframe['ED_R1_Tmax'].values.tolist())
    print(T_list)
    '''
    plt.figure()
    for i in range(len(T_list)):
        plt.plot(np.arange(len(T_list[i])), T_list[i], label=labels[i])
    plt.xlabel("time")
    plt.ylabel("MSE of total links delay during training")
    plt.legend()
    plt.savefig("%s_%s_%s.png" %(location, year,Tempreture))
    plt.close()
    '''
def print_tempreture(df, year, MAX_MIN):
    plt.figure()
    for location in ['Keis', 'Mila', 'Port', 'Rohw', 'St J', 'Ston', 'Vero']:
        TMAX_MIN = df.loc[(df_temp['YEAR'] == year) & (df['Location'] == location)][MAX_MIN].values.tolist()
        #print(df.loc[(df_temp['YEAR'] == year) & (df['Location'] == location) & (df['TMAX']>100)])
        for i in range(len(TMAX_MIN)):
            if TMAX_MIN[i] ==-99:
                TMAX_MIN[i]=-1
        plt.plot(np.arange(1, len(TMAX_MIN) + 1), TMAX_MIN, label=location)
    plt.xlabel("day")
    plt.ylabel(MAX_MIN)
    plt.legend()
    plt.savefig("%s_%s_with_out_Cst_Colu.png" % (year, MAX_MIN))
    plt.close()
def plot_count(df, count_feature, group_feature):
    group_list = df[group_feature].drop_duplicates()
    for group in group_list:
        df_group_feature = df.loc[(df[group_feature] == group)]
        df_group_feature[count_feature].value_counts().sort_index().plot(kind='bar')
        plt.xlabel(count_feature)
        plt.ylabel("# of entries")
        print(group)
        plt.legend(str(group))
        plt.show()

def plot_hist(df, feature):
    histogram = df[feature].plot.hist()
    print(histogram)
    plt.legend(feature)
    plt.savefig("plots/Data_statistics/hist/%s_hist.png" %(feature))
    plt.close()




#def find_tempreture
df_Origin = pd.read_csv('ML_Data_09162022.csv')
df_Aggre=pd.read_excel('ML_DB_V1.xlsx')
df_temp=pd.read_csv("WD_2012_14.csv")
print(df_Origin.columns)
years=df_Origin['YEAR'].unique().tolist()
print(years)
print(find_locations_of_each_year(df_Origin,2012))
print(find_locations_of_each_year(df_Origin,2013))
print(find_locations_of_each_year(df_Origin,2014))
locations=find_locations_of_each_year(df_Origin,2012)['Location'].values.tolist()
print(locations)
#print(find_PD_of_each_year_Location(df, 2012, 'C St'))

print_tempreture(df_temp, 2013, 'TMIN')

'''
for year in years:
    for location in locations:
        PD_year_location=find_PD_of_each_year_Location(df, year, location)
        print(PD_year_location)
'''


#print(df_Origin['Variety'].drop_duplicates().values.tolist())
#find_tempreture_in_differnt_year_differntlocation_differnt_period(df_Origin,'C St', 2012, 'ED_R1_Tmax')
'''
for location in locations:
    print(location)
    for year in [2012, 2013, 2014]:
        print(year)
        find_tempreture_in_differnt_year_differntlocation_differnt_period(df_Origin, location, year, "ED_R1_Tmax")
'''
locations_temp=df_temp['Location'].unique().tolist()
print(locations_temp)





'''
for location in locations:
    
    print(df_temp.loc[(df_temp['YEAR']==2012) & (df_temp['Location']==location)]['date'].values.tolist())
    print(df_temp.loc[(df_temp['YEAR']==2013) & (df_temp['Location']==location)]['date'].values.tolist())
    print(df_temp.loc[(df_temp['YEAR']==2014) & (df_temp['Location']==location)]['date'].values.tolist())
'''
'''
print(find_Variety_of_each_year_PD(df_Origin,2012, 'Keis', 1)['Variety'].values.tolist())
print(find_Variety_of_each_year_PD(df_Origin,2012, 'Keis', 2)['Variety'].values.tolist())
print(find_Variety_of_each_year_PD(df_Origin,2012, 'Keis', 3)['Variety'].values.tolist())
print(find_Variety_of_each_year_PD(df_Origin,2012, 'Keis', 4)['Variety'].values.tolist())

print(find_Variety_of_each_year_PD(df_Origin,2013, 'Keis', 1)['Variety'].values.tolist())
print(find_Variety_of_each_year_PD(df_Origin,2013, 'Keis', 2)['Variety'].values.tolist())
print(find_Variety_of_each_year_PD(df_Origin,2013, 'Keis', 3)['Variety'].values.tolist())
print(find_Variety_of_each_year_PD(df_Origin,2013, 'Keis', 4)['Variety'].values.tolist())

print(find_Variety_of_each_year_PD(df_Origin,2014, 'Keis', 1)['Variety'].values.tolist())
print(find_Variety_of_each_year_PD(df_Origin,2014, 'Keis', 2)['Variety'].values.tolist())
print(find_Variety_of_each_year_PD(df_Origin,2014, 'Keis', 3)['Variety'].values.tolist())
print(find_Variety_of_each_year_PD(df_Origin,2014, 'Keis', 4)['Variety'].values.tolist())


#find_Variety_of_each_location_all_year_all_period(df_Origin,'Vero')
#find_Variety_of_differrent_location_in_the_same_year_in_the_same_period(df_Origin)

print("period 1")
find_Variety_of_same_location_same_period_but_differnt_year(df_Origin, 'C St', 1)
print("period 2")
find_Variety_of_same_location_same_period_but_differnt_year(df_Origin, 'C St', 2)
print("period 3")
find_Variety_of_same_location_same_period_but_differnt_year(df_Origin, 'C St', 3)
'''
#print(plot_MG_count(df_Aggre, "MG"))
#print(plot_MG_count(df_Origin, "rMG1"))
#print((df_Origin.loc[df_Origin["rMG1"]==7])["rMG"])
#plot_count(df_Origin,"rMG", "PD")
#plot_count(df_Origin,"rMG", "Location")

#find the min and max value for yields, protain, oil, concentration
print("Yield max: %f" %(df_Origin.Yield.max()))
print("Yield min: %f" %(df_Origin.Yield.min()))
print("Oil max: %f" %(df_Origin.Oil.max()))
print("Oil min: %f" %(df_Origin.Oil.min()))
print("Protein max: %f" %(df_Origin.Protein.max()))
print("Protein max: %f" %(df_Origin.Protein.min()))
#plot_hist(df_Origin,"Yield")
#plot_hist(df_Origin,"Oil")
#plot_hist(df_Origin,"Protein")
feature_list=['MAX_Tmax_ED_R1', 'MIN_Tmax_ED_R1', 'SD_Tmax_ED_R1',
       'MAX_Tmin_ED_R1', 'MIN_Tmin_ED_R1', 'SD_Tmin_ED_R1', 'MAX_Srad_ED_R1',
       'MIN_Srad_ED_R1', 'SD_Srad_ED_R1', 'MAX_Phot_ED_R1', 'MIN_Phot_ED_R1',
       'SD_Phot_ED_R1', 'MAX_Tmax_R1_R5', 'MIN_Tmax_R1_R5', 'SD_Tmax_R1_R5',
       'MAX_Tmin_R1_R5', 'MIN_Tmin_R1_R5', 'SD_Tmin_R1_R5', 'MAX_Srad_R1_R5',
       'MIN_Srad_R1_R5', 'SD_Srad_R1_R5', 'MAX_Phot_R1_R5', 'MIN_Phot_R1_R5',
       'SD_Phot_R1_R5', 'MAX_Tmax_R5_R7', 'MIN_Tmax_R5_R7', 'SD_Tmax_R5_R7',
       'MAX_Tmin_R5_R7', 'MIN_Tmin_R5_R7', 'SD_Tmin_R5_R7', 'MAX_Srad_R5_R7',
       'MIN_Srad_R5_R7', 'SD_Srad_R5_R7', 'MAX_Phot_R5_R7', 'MIN_Phot_R5_R7',
       'SD_Phot_R5_R7']


#for feature in feature_list:
#    plot_hist(df_Origin, feature)
plot_hist(df_Origin,"PDOY")
#create the bins
#plot the hist plot for each of them, see the distribution

