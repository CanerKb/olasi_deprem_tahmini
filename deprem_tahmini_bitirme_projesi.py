################# VERİ SETİ HİKAYESİ ##################################

"""
Degıskenler ne anlama geliyor?
#ID: atadığım değer, olay sırasını gösterir.
#Kod: [YYYYMMDDHHMMSS (YearMonthDayHourDakikaSaniye)] olayının tek kimliği.
#Tarih: YYYY.AA.GG (Yıl.Ay.Gün) biçiminde belirtilen olay tarihi.
#Saat: Aşağıdaki biçimde belirtilen olayın başlangıç ​​zamanı (UTC) SS:DD:SS.MS (Saat:Dakika:Saniye.Milisaniye).
#Latitude, Longtitude: Etkinliğin koordinasyonları. (ondalık derece cinsinden)
#Depth(km): Etkinliğin kilometre cinsinden derinliği. 300 km'den daha az derinliğe sahip depremler 'yakın' olarak kabul edilir.
#xM: Belirtilen büyüklük değerlerinde (MD, ML, Mw, Ms ve Mb) en büyük büyüklük değeri.
#MD ML Mw Ms Mb: Büyüklük türleri (MD: Süre, ML: Yerel, Mw: Moment, Ms: Yüzey dalgası, Mb: Vücut dalgası).
#0.0 (sıfır), bu tür bir büyüklük için hesaplama yapılmadığı anlamına gelir.
#Tip: Deprem (Ke) veya Şüpheli Patlama (Sm).

#Büyüklük türleri hakkında daha fazla bilgi için lütfen kontrol edin: http://www.koeri.boun.edu.tr/bilgi/buyukluk.htm
"""

# gerekli kutuphanelerin import edilmesi
import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve

# !pip install xgboost
# !pip install catboost
# !pip install lightgbm
# !pip install sklearn

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# bazi gorsel ayarlarin yapilmasi
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# bazi hatalari almamak adina....
# from pandas.core.common import SettingWithCopyWarning
# from sklearn.exceptions import ConvergenceWarning

# veri setlerının python'na indirilmesi
df1 = pd.read_csv('olasi_deprem_tahmini/DATA/2000-2007.txt', delimiter = "\t", encoding='mbcs', header=0,names=["ID","Code","Date","Time","Latitude","Longtitude","Depth(KM)","xM","MD","ML","Mw","Ms","Mb","Type","Location"])
df2 = pd.read_csv('olasi_deprem_tahmini/DATA/2007-2010.txt', delimiter = "\t", encoding='mbcs', header=0,names=["ID","Code","Date","Time","Latitude","Longtitude","Depth(KM)","xM","MD","ML","Mw","Ms","Mb","Type","Location"])
df3 = pd.read_csv('olasi_deprem_tahmini/DATA/2010-2013.txt', delimiter = "\t", encoding='mbcs', header=0,names=["ID","Code","Date","Time","Latitude","Longtitude","Depth(KM)","xM","MD","ML","Mw","Ms","Mb","Type","Location"])
df4 = pd.read_csv('olasi_deprem_tahmini/DATA/2013-2015.txt', delimiter = "\t", encoding='mbcs', header=0,names=["ID","Code","Date","Time","Latitude","Longtitude","Depth(KM)","xM","MD","ML","Mw","Ms","Mb","Type","Location"])
df5 = pd.read_csv('olasi_deprem_tahmini/DATA/2015-2020.txt', delimiter = "\t", encoding='mbcs', header=0,names=["ID","Code","Date","Time","Latitude","Longtitude","Depth(KM)","xM","MD","ML","Mw","Ms","Mb","Type","Location"])

# slice ile concat icin listenin olusturulmasi
frames = [df1,df2,df3,df4,df5]
data = pd.concat(frames, names=["ID","Code","Date","Time","Latitude","Longtitude","Depth(KM)","xM","MD","ML","Mw","Ms","Mb","Type","Location"] )

#Veri setinin incelenmesi

data.info()

# veri setinin istatistiksel gozlenmesi
data.describe().T

# bu fonksiyon ile veri setinin on incelemeleri yapilir.
def check_df(dataframe, head=5):
    print("##################### İnfo #####################")
    print(dataframe.info())

    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum().sort_values(ascending=False))

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.98, 1]).T)

check_df(data)

# veri setinin tarih araligini check ediyorum.
data["Depth(KM)"].max()
# Out: 180.8
data.head()
# Makine ogrenmesi ve Veri Analizinde ise yaramayacak olan ID degiskeninin Silinmesi (Yeniden tanimlayacagim)
df_ = data.drop(['ID'], axis=1)

# tüm değişkenleri küçük harflerle temsil etmek.
# boylelikle kod yazmak daha hizli ve hatasiz olabilecek.
df_.columns = [col.upper() for col in df_.columns]

# Yeniden ID degiskeninin tanimlanmasi
df_.insert(loc= 0, column= 'ID', value= range(0, len(data)))
# ilk 5 satirinin gozlemlenmesi
df_.head()
# son 5 satirinin gozlemlenmesi
df_.tail()
"""
                        # Feature Engineer'in icin Not. 
                        # 300 km'den daha az derinliğe sahip depremler 'yakın' olarak kabul edilir.
                        
bu bilgiyi değişken oalrak kullanamayacağım anlamına geliyor: df_["DEPTH(KM)"].max() # Out: 180.8
Çünkü hepsi 300 km' den küçük
"""
# 2'den kucuk olan depremlerin adetinin hesaplanmasi icin ön kod denemesi
df_.shape[0] - df_.loc[df_["XM"] <= 2].shape[0]

len(df_.loc[df_["XM"] <= 2])

# ön kod denemesi sonrasi tum buyukluk degiskenlerinde for dongusu olusturarak 2'den kucuk buyukluklerin adetine ulasmak.
# buyukluklari liste icinde atamak
magnitud = ["MD", "MS", "MB", "MW", "ML", "XM"]

for col in magnitud:
    print(col + "<=2 : ", len(df_.loc[df_[col] <= 2]))
    print("*******************************")


"""
                            Feature Engineer yapmak icin bir fikir daha.
# oluşum yılını yeni bir sütun olarak eklemek.
    # df.insert(loc= 3, column= "Year", value= data["Date"].str[0:4], True)
"""
# http://www.koeri.boun.edu.tr/bilgi/buyukluk.htm
# bazı notlar:

# MARMARA Bölgesinin verisetinde seçilmesi
mr = '"EDİRNE"|"EDIRNE"|"KIRKLARELI"|"KİRKLARELİ"|"KIRKLARELİ"|"KİRKLARELI"|"TEKIRDAG"|"TEKİRDAG"|"TEKİRDAĞ"|"TEKIRDAĞ"|"ISTANBUL"|"İSTANBUL"|"KOCAELI"|"KOCAELİ"|"SAKARYA"|"BILECIK"|"BİLECİK"|"BILECİK"|"BİLECIK"|"YALOVA"|"BURSA"|"BALIKESIR"|"BALİKESİR"|"BALIKESİR"|"ÇANAKKALE"|"CANAKKALE"'
mr= mr.replace("|", ",")
mr

mr = mr.replace('","', '|')
mr

mr= mr.replace('"', '')
mr

marmara = mr

check_df(df_)
# dfm adinda marmara bolgesi veri setinin olusturulmsi
dfm= df_[df_['LOCATION'].str.contains(marmara,na=False, case= False)]
dfm.head()
check_df(dfm)

# zamanla gerçekleşen depremleri zaman sırasında takip edebilmek için DATE & TIME a göre sıralanır.
dfm.sort_values(by= ["DATE","TIME"],ascending= True, axis= 0, inplace= True, ignore_index= True)
dfm.index = np.arange(0, len(dfm))
dfm.head(20)
dfm["DEPTH(KM)"].max()
# Out[218]: 102.0

check_df(dfm)

# toplam verı setı gözlem sayım
dfm.shape[0]
# Out[232]: 19008


sayi= len(dfm.loc[dfm["MD"] == 0])
oran= len(dfm.loc[dfm["MD"] == 0])/dfm.shape[0] * 100
print("MD 'deki sıfır olan gözlem sayısı: {} \nMD'nin veri içindeki sıfır oranı: {:.3f}".format(sayi, oran))
# MD 'deki sıfır olan gözlem sayısı: 9439
# MD'nin veri içindeki sıfır oranı: 49.658

# bu durum bir döngü doğurmakta:
magnitud = ["MD", "MS", "MB", "MW", "ML", "XM"]

for col in magnitud:
    sayi= len(dfm.loc[dfm[col] == 0])
    oran= len(dfm.loc[dfm[col] == 0])/dfm.shape[0] * 100
    print(col+"'deki sıfır olan gözlem sayısı: {} ".format(sayi))
    print(col+"'nin veri içindeki sıfır oranı: {:.3f}".format(oran))
    print("*****************************************************", end= "\n")

# Bu sıfırların yapısal oldugunu verı setı hıkayesınden bılıyoruz.
# msno kütüphanesi ile görselleştırebılmemız için 0 'lar yerıne NaN koymalıyız.

dfm_1= dfm
magnitud = ["MD", "MS", "MB", "MW", "ML", "XM"]

# veri setinde 0 olan deprem büyüklükleri aslında ölçülmeyen değerlerdi. Onları None ifade haline getiriyorum.
dfm_1.replace(0.0, None, inplace=True)
dfm_1.isnull().sum().sort_values(ascending= False)
dfm.isnull().sum().sort_values(ascending= False)


# her değişiklik sonrası check etmekte fayda var. Gözden kaçan bir durum var ı ve verı setı setı hakımıyetını
    # canlı tutmak ıcın
check_df(dfm_1)

# burada yapısal ya da rassallığı gözlemliyoruz. ÖNEMLİ
# msno kutuphanesinin import edilmesi
import missingno as msno
none_deger_iliskisi = msno.matrix(dfm_1), plt.show(block= True)
msno.bar(dfm_1), plt.show(block= True)
# Grafige gore rassal bir eksiklik degil, yapısal bir eksiklikten bahsedebiliriz.

# none_deger_iliskisi grafiği incelendiğinde "MD" ile "ML" değişkenleri ayrı zamanlarda ayrı ayrı
    # ölçüldüğü görülmektredir. Buradan bir değer ile ölçüldükten sonra diğer ölçümün yapılmadığı düşüncesiyle bu iki
    # değeri birleştireceğiz.

# XM maksımum buyuklugu alıyordu.
# o zaman none_deger_iliskisi grafik incelendiğinde MD'nin sıfırdan farklı oldugu değerleri XM direk almalı.
# Çünku baska buyukluk yok!

# veriseti içinde bu durumu gözlemlemek
dfm_1.loc[(dfm_1["MD"] >= 0 ) , ["XM", "MD"]].head(20)

# none_deger_iliskisi grafikte bazı durumlarda MD ve ML 'nin birlikte hesaplandıgı durumlar var gıbı durmakta
# onları ınceleyelım.
dfm_1.loc[((dfm_1["MD"] >= 0) & (dfm_1["ML"] >= 0)) , ["XM", "MD", "ML"]].head(20)
len(dfm_1.loc[((dfm_1["MD"] >= 0) & (dfm_1["ML"] >= 0)) , ["XM", "MD", "ML"]])
# Out[368]: 102
# 102 adet gözlem için ML ve MD değişkenleri hesaplanmıştır! aralarından büyük olan XM e yazılmıstır.


# bu durumu biraz daha yakından inceleyelim.
MD_ML_df = dfm_1.loc[((dfm_1["MD"] >= 0) & (dfm_1["ML"] >= 0)) , ["XM", "MD", "ML"]]

MD_ML_df["MD-ML"] = abs(MD_ML_df["MD"] - MD_ML_df["ML"])

MD_ML_df.head(20)

MD_ML_df["MD-ML"].agg(["min", "max", "mean"])
# Out[378]:
# min    0.000
# max    0.700
# mean   0.114

# deprem kuvvetleri log 10 tabanında değişim gösterirler. Buradaki her 1'lik değişim gerçekte ~10'un katı olacak şekilde
# değişime uğramış olur. burada ortalama 1 olarak baz alındıgında ~0.114 'lük bir değişimin ~1.3'lik bir değişimdir.
# Bu etki göz önüne alınabilir ve bu iki büyüklük değerinin ortalaması veri setine yansıtılabilir.
# ya da verisetin deki oran dikkate alındıgında;
print("Yüzde : ",((len(dfm_1.loc[((dfm_1["MD"] >= 0) & (dfm_1["ML"] >= 0))]))/dfm_1.shape[0]) *100)
# Yüzde :  0.5366161616161615
    # verisetinin %1'i bile değil.
    # bu yüzden önemsiz olarak da işlem görmeden devam edilebilir.

dfm_1.head()
kesisim_MD_ML = dfm_1.loc[((dfm_1["MD"] >= 0) & (dfm_1["ML"] >= 0)) , ["MD", "ML"]]
dfm_1.loc[((dfm_1["MD"] >= 0) & (dfm_1["ML"] >= 0)) , ["MD", "ML"]]


# XM e MD ve ML 'nin kesişimlerindeki ortalama değerini atamak istiyorum. Python yeteneklerim gelişsin.
kesisim_MD_ML["ML_MD_ort"] = (kesisim_MD_ML["ML"] + kesisim_MD_ML["MD"]) / 2

dfm_1.head()

dfm_1["XM_Duzeltme"] = kesisim_MD_ML["ML_MD_ort"]
# ML_MD_Series = pd.Series(kesisim_MD_ML["ML_MD_ort"])

# dfm_1.loc[dfm_1["XM_Duzeltme"] == ML_MD_Series , "XM" ]


dfm_1["XM"][9241]
# Out[103]: 2.5
dfm_1["XM_Duzeltme"][9241]
# Out[106]: 2.45

indeks = kesisim_MD_ML["ML_MD_ort"].index

dfm_1["XM"][indeks] = kesisim_MD_ML["ML_MD_ort"]

dfm_1["XM"][9241]
# Out[106]: 2.45  oldu düzeldi

# dfm_1 içerisinde XM değişkenine ML ve MD  değerlerinin kesişimlerindeki değerlerin ortalamasını atamış olduk.

# dfm_1 ile XM değişkenini büyüklük anlamında ortalamayı daha iyi temsil eder hale getirmiş olduk!
# Biliyoruz ki artık XM dışında bir Büyüklük değişkenine ihtiyacımız bulunmamakta.


check_df(dfm_1)
##################### Types #####################
# ID               int64
# CODE             int64
# DATE            object
# TIME            object
# LATITUDE       float64
# LONGTITUDE     float64
# DEPTH(KM)       object
# XM              object
# MD              object
# ML              object
# MW              object
# MS              object
# MB              object
# TYPE            object
# LOCATION        object
# XM_Duzeltme     object
# dtype: object




# Değişken Tiplerinin düzeltilmesi

dfm_1[["XM", "DEPTH(KM)"]]= dfm_1[["XM", "DEPTH(KM)"]].astype("float64")
dfm_1.dtypes

dfm_1["DATE_TIME"] = dfm_1["DATE"] + " " + dfm_1["TIME"]
dfm_1.head()
dfm_1["DATE_TIME"].tail()

dfm_1["DEPTH(KM)"].max()

dfm_1["DATE_TIME"] = pd.to_datetime(dfm_1["DATE_TIME"])
dfm_1["DATE"] = pd.to_datetime(dfm_1["DATE"])
# dfm_1["TIME"] = pd.to_datetime(dfm_1["TIME"], format="%H:%M:%S", unit="")

# dfm_1["TIME"].drop(inplace=True, axis=1)

check_df(dfm_1)
##################### Types #####################
# ID                      int64
# CODE                    int64
# DATE           datetime64[ns]
# TIME           datetime64[ns]
# LATITUDE              float64
# LONGTITUDE            float64
# DEPTH(KM)             float64
# XM                    float64
                                            # SİLİNECEKLER!
                                    # MD                     object
                                    # ML                     object
                                    # MW                     object
                                    # MS                     object
                                    # MB                     object
                                    # TYPE                   object
                                    # LOCATION               object
                                    # XM_Duzeltme            object
# DATE_TIME      datetime64[ns]
# dtype: object


dfm_1["LOCATION"].nunique()
# Out[128]: 13996


dfm_1.insert(3,"HOUR", dfm_1["TIME"].str[0:2])
dfm_1.head()

dfm_1["HOUR"].nunique()
#24

check_df(dfm_1)

# deprem buyuklugununun 2 den kucuk olma durumunun tekrar incelenmesi
for col in magnitud:
    print(col + "<=2 : ", len(dfm_1.loc[dfm_1[col] <= 2]))
    print(col + " %  : ", (len(dfm_1.loc[dfm_1[col] <= 2])/len(dfm_1) * 100))
    print("*******************************")

# veri setinden çıkarılacak değişkenler: büyükler (XM dışındakiler) ve XM_Duzeltme, TYPE ve LOCATION değişkenleri
df_drop = ["MS", "MB", "MW", "ML", "MD", "TYPE", "XM_Duzeltme", "LOCATION", "TIME"]

dfm_1.drop(labels=df_drop, axis=1, inplace= True)

dfm_1.head(20)
dfm_1.isnull().sum()
# DEPTH(KM)     488 boş hücre var
dfm_1.shape
# Out[98]: (19008, 9)

# siliyorum
dfm_1.dropna(inplace=True)


dfm_1["DATE_TIME"].min() # Out: Timestamp('2000-01-01 05:59:53.200000')


dfm_1["DATE_TIME"].max() # Out: Timestamp('2019-12-31 02:41:57.820000')

# enlemdeki çeşitliliğe bakıyor marmara bölgesi olarak sınırladıgımız ıcın acaba bır feature engineer çıkar mı!?
dfm_1["LATITUDE"].nunique()
# Out[102]: 7358   ÇIKMADI!


# tarih değişkeninden yeni değişkenler üretmek:
# Deprem ile bilimsel alakası sadece yer kabugu altındakı hareketliliğin ve basınc vs bırıkımının zamanı bir şekilde
# içeriyor olması.
def create_date_features(df, date_column):
    df['month'] = df[date_column].dt.month
    df['day_of_month'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.weekofyear
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['year'] = df[date_column].dt.year
    df["is_wknd"] = df[date_column].dt.weekday // 4
    return df

df_new = create_date_features(dfm_1, "DATE_TIME")

df_new.head()
df_new.shape[0]
# derinlikte değişim olacak mı diye ayrıca sürekli takip etmekteyim verisetim içinde önemli bir değişken şimdilik
df_new["DEPTH(KM)"].max()
# Out[109]: 102.0

"""
                                    # Deprem sınıfları 
                                        Kaynak:
http://www.koeri.boun.edu.tr/sismo/bilgi/depremnedir/index.htm#:~:text=Bu%20s%C4%B1n%C4%B1fland%C4%B1rma%20tektonik%20depremler%20i%C3%A7in,den%20fazla%20derinli%C4%9Finde%20olan%20depremlerdir.

sığ deprem                  : 0-60(70) km 
orta derinlikteki deprem    : 70- 300 km
derin deprem                : >300 

"""
df= df_new



                    # anlamlı bir sınıflandırma çıkmamakta!
# df.loc[(df["DEPTH(KM)"] < 60), "DEPTH(KM)_SINIF"] = "SIG_DEPREM"        # HEPSİ SIG_DEPREM (2 adet hariç)
# df.loc[(df["DEPTH(KM)"] >= 70), "DEPTH(KM)_SINIF"] = "SIG_ORTA_DEPREM"  # 2 adet
# df.loc[(df["DEPTH(KM)"] > 300), "DEPTH(KM)_SINIF"] = "DERIN_DEPREM"     # BÖYLE BİR SINIF OLUŞAMAYACAK
# df.drop(labels="DEPTH(KM)_SINIF", axis=1, inplace= True)

# model öncesi çıkarılacaklar: "ID","CODE","DATE","DATE_TIME"
# df.drop(labels=["ID","CODE","DATE","DATE_TIME"], axis=1, inplace=True)
df.head(25)
df["DEPTH(KM)"].mean()
df["DEPTH(KM)"].max()
df["DEPTH(KM)"].min()


df["XM"].shape[0]
# Out[116]: 18520

df["XM"][17520:18520]

                                    # Grafikler İçin


plt.plot(df["DATE"][17520:18520],df["XM"][17520:18520],'r-')
plt.title("Marmara Bölgesi")
plt.ylabel("XM")
plt.xlabel("DATE")
plt.show(block= True)

plt.plot(df["LONGTITUDE"][17520:18520],df["XM"][17520:18520],'c-')
plt.title("Marmara Bölgesi")
plt.ylabel("LONGTITUDE")
plt.xlabel("DATE")
plt.show(block= True)

plt.plot(df["DATE_TIME"][17520:18520],df["XM"][17520:18520],'bo-')
plt.title("Marmara Bölgesi")
plt.xlabel("DATE_TIME")
plt.ylabel("XM")
plt.show(block= True)


# boylamları sıralayıp o şekilde üzerlerinde gerçekleşen depremleri görmek ıstedım.
df_LONGTITUDE = df.sort_values(by= "LONGTITUDE", ascending=True)

plt.plot(df_LONGTITUDE["LONGTITUDE"][1000:],df_LONGTITUDE["XM"][1000:],'go--')
plt.title("Marmara Bölgesi")
plt.ylabel("XM")
plt.xlabel("LONGTITUDE")
plt.show(block= True)

sns.boxplot(data= df, x="XM"), plt.show(block= True)

sns.boxenplot(data= df, x="XM"), plt.show(block= True)

# sns.barplot(data= df, x="LONGTITUDE", y="XM"), plt.show(block= True)  # kesikli gibi davranıyor LONGTITUDE



################################################################
# Gerekli Fonksiyonlarin Tanimlanmasi
################################################################

# Aykırı Değerlerin Tespt Edilmesi
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı Değer Sorgusu
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Aykırı Değer Baskılama
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değerleri Gözlemlemek. Aykırılıkların index bilgisi ve 10 eşik degerınde aykırılıkları gozlemlemek.
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    # alt ve üst aykırı degerler 10 dan buyukse ilk 5 gözlemi görmek ıstıyorum.
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    # alt ve ust aykırı degerler 10dan kucukse direkt hepsini görmek ıstıyorum.
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    # index argümanını True yaparsam index bilgilerini almak ıstıyorum dıyorum.
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index



# Değişkenlerin Kategorize Edilmesi
# cat_th=13 : MONTH değişkeninden ötürü
# car_th=25 : HOUR değişkeninden ötürü
def grab_col_names(dataframe, cat_th=13, car_th=25):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    # cat_cols: df kolonlarinda gez tipi object olanlar kategoriktir.
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    # numerik fakat kategorik: df kolonlarinda gez essiz sinif sayisi cat_th den kucuk olanlar kategoriktir ve
    # ayni zamanda tipi object olmayanlar: zaten yukarda objectleri aldik burada amac sayisal siniflandirmayi yakalamak.
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    # categorik ama kardinal: df kolonlarinda gez essiz sinif sayisi car_th den buyuk olanlar kardinaldir bunu yaparken
    # ayni zamanda kolon dtype object olanlarda yap.
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    # cat_cols u guncellemek lazim: numerik gorunumlu ama kategorik olanlari cat_cols a ekleriz ve cardinal'leri
    # ondan cikaririz.
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    # num_cols: df kolonlarinda gez ve tipi object olmayanlari listele.
    # yalniz burada date/tarih'ler de gelecektir sonradan cikartabiliriz. Manuel olarak.
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

check_df(df)

# tüm değişkenleri küçük harflerle temsil etmek.
df.columns = [col.upper() for col in df.columns]

# df["HOUR"]= df["HOUR"].astype("float64")
# df["HOUR"].dtypes
df["WEEK_OF_YEAR"].nunique()
df["DAY_OF_YEAR"].nunique()
df["DAY_OF_MONTH"].nunique()
df["MONTH"].nunique()


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# cat_th=13, car_th=25 iken değişkenlerimin sınıflandırılması : Burada HOUR'dan ötürü kardinal alt sınırı 25 yapıyorum.
# Observations: 18520
# Variables: 16
# cat_cols: 3
# num_cols: 13
# cat_but_car: 0
# num_but_cat: 3


# num_cols
# Out[159]:
# ['ID',
#  'CODE',
#  'DATE',
#  'LATITUDE',
#  'LONGTITUDE',
#  'DEPTH(KM)',
#  'XM',
#  'DATE_TIME',
#  'DAY_OF_MONTH',
#  'DAY_OF_YEAR',
#  'WEEK_OF_YEAR',
#  'YEAR']

            # cat_cols
            # Out[473]:
# ['MONTH', 'DAY_OF_WEEK', 'IS_WKND']

num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]

# num_cols
# Out[161]:
# ['LATITUDE',
#  'LONGTITUDE',
#  'DEPTH(KM)',
#  'XM',
#  'DAY_OF_MONTH',
#  'DAY_OF_YEAR',
#  'WEEK_OF_YEAR',
#  'YEAR']

#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

#############################################
# 1. Kesifci Değişken Analizi (EDA)
#############################################


# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


#############################################

# 1. Genel Resim

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum().sort_values(ascending=False))

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.98, 1]).T)

check_df(df)

#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#############################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]


# kategorik degiskenlerin frekans ve yuzdece degerlerini gozlemlemek
def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block= True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]


#############################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#############################################

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=25)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block= True)

num_summary(df, num_cols, plot= True)

# sağdan çarpıklar: log-transform dönüşümü uygulanır   (grafiğin kendi sağı ve solu için)
        # LONGTITUDE, DEPTH(KM), XM
# soldan çarpıklar: yeo-jehson dönüşümü uygulanır.
        # LATITUDE
# Bu dönüşümler normal dağılıma veriyi uygurmak içindir çünkü yapacagımız işlemler bu kabule dayanmaktadır.

#############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#############################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(dataframe.groupby(categorical_col).agg({target : ["mean","count"]}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "XM", col)




#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################

def high_correlated_cols(dataframe, plot=False, corr_th=0.92):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [upper_triangle_matrix.loc[(upper_triangle_matrix[col] > corr_th) , col] for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu", annot= True, linewidths=.5)
        plt.show(block= True)
    return drop_list

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]

df_corr = cat_cols + num_cols

drop_list = high_correlated_cols(df[df_corr], plot=True, corr_th=0.92)
drop_list

# aşağıdaki çıktıda da görüleceği üzere yüksek korelasyonlar mevcut

# değişkenler birbirleri içinden türemektedir. Yüksek kombinasyonlara ve farklılaşmaya sahip olmadıklarından
# korelasyonlarının iyi ve yüksek çıkması beklenti dahilindedir.

# Out[504]:

# [MONTH   0.997                       # ~1
#  Name: DAY_OF_YEAR, dtype: float64,

#  MONTH         0.972
#  DAY_OF_YEAR   0.974
#  Name: WEEK_OF_YEAR, dtype: float64]

# bu degiskenleri ya silerim ya da feature engineer de kullanirim. Ki feature engineer de kullanmak en iyisi.
# sonrasında silerim. DAY_OF_YEAR'ı


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################


#############################################
# 1. Missing Values (Eksik Değerler) : KNn ile doldurmak
#############################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)
# eksik değerim bulunmamakta!

#############################################
# LOF ile cok degiskenli aykiri deger analizi yapmak
#############################################
"""
# df= df_new
"""
df[["MONTH", "DAY_OF_WEEK", "IS_WKND"]]= df[["MONTH", "DAY_OF_WEEK", "IS_WKND"]].astype("O")
df.dtypes
# cat_cols ve num_cols değişkenlerim değişti ise yeniden tanimlamak gerekir.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]

# dfff = df[cat_cols + num_cols]
# one_h = cat_cols + num_cols
# df[one_h].dtypes
# öncelikle lof için uzaklık tabanlı algoritma çalışacagı için ENCODER işlemi yapmamız lazım.
df = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
# df_e = pd.get_dummies(dfff[one_h], drop_first=True)
# df_e.head()
# LOF u import ediyorum
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores = -df_scores
np.sort(df_scores)[0:5]

# sıraladığımız skorları DataFrame cevırıp score adlı değişkene atadık. dataframe e cevirme amacimiz grafik icin.
scores = pd.DataFrame(np.sort(df_scores))
# scores adlı DataFrame grafiğini çiziyoruz. Burada xlim ile oynayıp en ıyı kırılmaya karar verebılırız.
scores.plot(stacked=True, xlim=[0, 50], style='.-'), plt.show(block=True)
# burada gorulecegi uzeri coklu gozlemlerde aykiri degerim bulunmamaktadir.

# df_new.dtypes
df.dtypes
df.head()
df= df_new



#############################################
# 2. Outliers (Aykırı Değerler) - degisken bazinda
#############################################
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]


# aykiri degerleri gozlemlemek
for col in num_cols:
    sns.boxplot(data= df, x= df[col])
    plt.show(block= True)

    # LATITUDE 1 adet min tarafı
    # LONGTITUDE 1 adet max tarafı
    # DEPTH(KM) max tarafı sayılamayacak duzeyde
    # XM max tarafı 24 ADET GİBİ DURUYOR

    # DİĞERLERİNDE DEĞİŞKEN BAZINDA AYKIRILIK GÖZLENMEMEKTE


# gozlemlenen aykiri degerleri check etmek
for col in num_cols:
    print(col, check_outlier(df, col))
# LATITUDE True
# LONGTITUDE True
# DEPTH(KM) True
# XM True
# DAY_OF_MONTH False
# DAY_OF_YEAR False
# WEEK_OF_YEAR False
# YEAR False

# AYKIRI DEĞERLERİN İNDEKSLERİNDEN YAKALANIP İNCELENMESİ GEREKMEKTEDİR. ŞİMDİLİK YAPMAYACAGIM!
# KARAR AĞACI ALGORITMALARIYLA CALIŞILACAKSA BU AYKIRI DEGERLERE HIC BIRSEY YAPILMASADA OLUR!

# gozlemlenen aykiri degerleri baskılamak
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# gozlemlenen aykiri degerleri check etmek
for col in num_cols:
    print(col, check_outlier(df, col))
# tüm aykırılıklar False oldu!

df.head()



#############################################
# Rare Encoding
#############################################

# RARE ile kategorik değişkenlerde azınlık&çoğunluk durumu kontrol edilir.


check_df(df)
df.head()
cat_summary(df, "DAY_OF_WEEK", plot=False)

df["DAY_OF_MONTH"].value_counts()
df.groupby("DAY_OF_MONTH")["XM"].mean()
# DAY_OF_MONTH için çeşitlilik 902 ila 299 arasında değişmektedir.
# ancak çıktı değişkeni üzerinde etkileri eşit oranda oldugu için ve çeşitlilik için belirli kırılımlar söz konusu
# olmadıgı için DAY_OF_MONTH bu değişken için RARE gerekli değildir.

df["DAY_OF_YEAR"].value_counts()
df.groupby("DAY_OF_YEAR")["XM"].mean()

df["WEEK_OF_YEAR"].value_counts()
df.groupby("WEEK_OF_YEAR")["XM"].mean()

col_rare_analys= ["DAY_OF_MONTH", "DAY_OF_YEAR", "WEEK_OF_YEAR"]

def rare_analyser(dataframe, target, col_rare_analys):
    for col in col_rare_analys:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "XM", col_rare_analys)


# rare yapmamayı secmış oluyorum.



#############################################
# 3. Feature Extraction (Özellik Çıkarımı)
#############################################

# %92'den fazla korelasyona sahip degiskenler uzerinden cikarimlar yapalim
df_corr = cat_cols + num_cols

drop_list = high_correlated_cols(df[df_corr], plot=True, corr_th=0.92)
drop_list
            # Out[357]:
            # [DAY_OF_YEAR   0.973
            #  Name: WEEK_OF_YEAR, dtype: float64,



df.head()

df[["MONTH","DAY_OF_YEAR"]]= df[["MONTH", "DAY_OF_YEAR"]].astype("float64")

# pay: yılın son ayından ne kadar uzaktayım / payda: yılın son gününden ne kadar uzaktayım.
df["MONTH_AND_DAY_OF_YEAR_"] = (13 - df["MONTH"]) / (367 - df["DAY_OF_YEAR"])

# pay: yılın son ayından ne kadar uzaktayım / payda: yılın son haftasından ne kadar uzaktayım.
df["MONTH_AND_WEEK_OF_YEAR_"] = (13 - df["MONTH"]) / (53 - df["WEEK_OF_YEAR"])

# pay: yılın son haftasından ne kadar uzaktayım / payda: yılın son gününden ne kadar uzaktayım.
df["DAY_OF_YEAR_AND_WEEK_OF_YEAR_"] = (53 - df["WEEK_OF_YEAR"]) / (367- df["DAY_OF_YEAR"])

# korelasyonu yüksek olanlar üzerinde feature engineer yaptım ancak daha fazla da türetilebilir. Ve iki anlamlılıkları
# test edilebilir. proportions_ztest testi ile

    # Ancak burada 2 sınıflı bir değer üretmiyoruz o yüzden yukarıda üretilen değişkenler için test yapamıyoruz!
# proportions_ztest import edelim
from statsmodels.stats.proportion import proportions_ztest
# proportions_ztest testi sunu ders iki degisken arasinda anlamli bir fark yoktur der. p<0.05 ise RED!
# Ki çıktı değişkenimizde continue veri oldugu için doğrudan göremeyiz!

# birim derinlik başına düşen deprem büyüklüğü
df["XM_AND_DEPTH(KM)_"] = df["XM"] / df["DEPTH(KM)"]

# Hissedilebilecek olan deprem sınıflandırılması
df.loc[(df["XM"] >= 3) & (df["DEPTH(KM)"] <= 70), "FEEL_QUAKE"] = "FEEL_QUAKE_YES"
df.loc[(df["XM"] < 3)  & (df["DEPTH(KM)"] <= 70), "FEEL_QUAKE"] = "FEEL_QUAKE_NO"
df.loc[(df["XM"] > 3)  & (df["DEPTH(KM)"] > 70), "FEEL_QUAKE"] = "FEEL_QUAKE_NO"
df["FEEL_QUAKE"].nunique()
df["FEEL_QUAKE"].unique()

df.head()

df.shape
df.dtypes

# 21 değişkene ulaşıldı! daha da artırılabılınır.


#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

    #############################################
    # One-Hot Encoding
    #############################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Observations: 18520
# Variables: 21
# cat_cols: 5
# num_cols: 16
# cat_but_car: 0
# num_but_cat: 1

num_cols= ['LATITUDE',
 'LONGTITUDE',
 'DEPTH(KM)',
 'XM',
 'DAY_OF_MONTH',
 'DAY_OF_YEAR',
 'MONTH_AND_DAY_OF_YEAR_',
 'MONTH_AND_WEEK_OF_YEAR_',
 'DAY_OF_YEAR_AND_WEEK_OF_YEAR_',
 'XM_AND_DEPTH(KM)_',
 'MONTH_AND_WEEK_OF_YEAR_']

num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]

df[num_cols].tail()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# Boş olan hücrelere-degerlere NaN lere de bir değer ataması yapmak için get_dummies içerisinden
    # dummy_na argümanını=True yaparız

# burada OHE yapacagım değişkenleri kendim seçiyorum
# HOUR değişkeninden ötürü üst sınır şartını 25 seçiyorum
ohe_cols = [col for col in df.columns if 25 >= df[col].nunique() >= 2]
# ohe_cols
# Out[381]: ['HOUR', 'MONTH', 'DAY_OF_WEEK', 'YEAR', 'IS_WKND', 'FEEL_QUAKE']


df = one_hot_encoder(df, ohe_cols, drop_first=True )
df.head()
df.shape

# 75+1 değişkene ulaşıldı! daha da artırılabılınır.

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]


# Observations: 18520
# Variables: 76
# cat_cols: 61
# num_cols: 15
# cat_but_car: 0
# num_but_cat: 61


#############################################
# Outliers (Aykırı Değerler)
#############################################
# Feature engineer işleminden sonra bir kez daha aykırılık analizi yapmakta fayda var.

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]


for col in num_cols:
    print(col, check_outlier(df, col))

# feature engineer ile ürettiğim değişkenlerde aykırılıklar mevcut
# MONTH_AND_DAY_OF_YEAR_ True
# MONTH_AND_WEEK_OF_YEAR_ True
# DAY_OF_YEAR_AND_WEEK_OF_YEAR_ True
# XM_AND_DEPTH(KM)_ True

# aykırı olan numerıc değişkenlerimi görselleştiriyorum
for col in num_cols:
    sns.boxplot(data=df, x= col)
    plt.show(block= True)

# Aykırı Değerleri Gözlemlemek. Aykırılıkların index bilgisi ve 10 eşik degerınde aykırılıkları gozlemlemek.
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    # alt ve üst aykırı degerler 10 dan buyukse ilk 5 gözlemi görmek ıstıyorum.
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())

    # alt ve ust aykırı degerler 10dan kucukse direkt hepsini görmek ıstıyorum.
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    # index argümanını True yaparsam index bilgilerini almak ıstıyorum dıyorum.
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

outlier_MONTH_AND_DAY_OF_YEAR_index = grab_outliers(df,"MONTH_AND_DAY_OF_YEAR_", True)
len(outlier_MONTH_AND_DAY_OF_YEAR_index)

# outlier'ları yakalamak Döngü içinde
outliers_col_indeks= []

for col in num_cols:
    outlier_index = grab_outliers(df, col, index=True)
    outliers_col_indeks.append(outlier_index)

    print(col, ": ", outlier_index, end="\n")
outliers_col_indeks

# bu aykırılıklar incelenmelidir. Ancak Vakten çok az bir sürem var ve Karar Ağaçları ile ilerleyeceğim için aykırı
# değerlerden etkilenmiyor oluşu karar ağaçlarının bu aykırılıklara bir şey yapmamayacagım anlamına gelebilmektedir.

df.isnull().sum()


# aykırı degerlerı baskılamak
for col in num_cols:
    replace_with_thresholds(df, col)


# aykırı değerler var olacaktır
for col in num_cols:
    print(col, check_outlier(df, col))



#############################################
# Korelasyon Analizi (Analysis of Correlation)

# Tekrardan - Son_corr
#############################################

def high_correlated_cols(dataframe, plot=False, corr_th=0.92):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [upper_triangle_matrix.loc[(upper_triangle_matrix[col] > corr_th) , col] for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu", annot= True, linewidths=.5)
        plt.show(block= True)
    return drop_list

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]

df_corr = cat_cols + num_cols

drop_list = high_correlated_cols(df[df_corr], plot=True, corr_th=0.92)
drop_list



# WEEK_OF_YEAR korelasyonu yüksek oldugu için siliyorum.
df.drop(labels="WEEK_OF_YEAR", axis=1, inplace=True)

#############################################
# 5. Feature Scaling (Özellik Ölçeklendirme)
#############################################

# df["YEAR_2003"].head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]
df.head()
df.tail()
"DAY_OF_MONTH", "DAY_OF_YEAR"

num_cols= ['LATITUDE', 'LONGTITUDE', 'DEPTH(KM)', 'XM', 'DAY_OF_MONTH', 'DAY_OF_YEAR', 'MONTH_AND_DAY_OF_YEAR_',
           'MONTH_AND_WEEK_OF_YEAR_', 'DAY_OF_YEAR_AND_WEEK_OF_YEAR_', 'XM_AND_DEPTH(KM)_', 'MONTH_AND_WEEK_OF_YEAR_',
           'DAY_OF_MONTH', 'DAY_OF_YEAR']

num_cols = [col for col in num_cols if col not in ["ID", "CODE", "DATE", "DATE_TIME"]]


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.tail()

# veri dönüşümü
from scipy import stats



df.dtypes

df.isnull().sum().sort_values(ascending=False)


# boş değerimiz yok ama yine de
df.dropna(inplace=True)



#############################################
# Base Models
#############################################
df.head()
# Hedef degiskenin atanmasi
y = df["XM"]
# Girdi Degiskenlerinin atanmasi
X = df.drop(["XM", "ID", "CODE", "DATE", "DATE_TIME"], axis=1)


X.head()
X.dropna(inplace=True)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]


# Modellerin Cross Validate metonu ile 10 katlı çapraz oluşturulması.
for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

df.head()

"""
df[num_cols] = pd.DataFrame(scaler.inverse_transform(df[num_cols]), columns= df[num_cols].columns).head()

df.head()
"""


# RMSE: 1885709694.1947 (LR)
# RMSE: 0.6826 (Ridge)
# RMSE: 1.0161 (Lasso)
# RMSE: 1.0161 (ElasticNet)
# RMSE: 0.889 (KNN)
# RMSE: 0.4331 (CART)
# RMSE: 0.3302 (RF)
# RMSE: 0.2604 (SVR)
# RMSE: 0.3702 (GBM)
# RMSE: 0.1837 (XGBoost)
# RMSE: 0.1971 (LightGBM)
# RMSE: 0.151 (CatBoost)


################################################
# LightGBM
################################################

lgbm_model = LGBMRegressor(random_state=18)

lgbm_params = {"learning_rate": [0.01, 0.1],
                "n_estimators": [300, 500],
                "colsample_bytree": [0.7, 1],
               "Num_leaves": [70, 80 ],
               "max_depth" : range(5, 11)}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_
# Out[785]:
# {'Num_leaves': 70,
#  'colsample_bytree': 0.7,
#  'learning_rate': 0.1,
#  'max_depth': 10,
#  'n_estimators': 500}
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse
# RMSE: 0.16386172232668386 (LightGBM)
# Önceki RMSE:  0.197 (LightGBM)

# model yeni hiperparametreleri ile daha başarılı bir sonuc elde etti


lgbm_model = LGBMRegressor(random_state=18)

lgbm_params2 = {"learning_rate": [0.085, 0.1, 0.15, 0.18],
                "n_estimators": [450, 500, 550, 650],
                "colsample_bytree": [0.9, 1, 1.5, 1.8],
               "Num_leaves": [60, 70 ],
               "max_depth" : range(3, 12)}

lgbm_best_grid2 = GridSearchCV(lgbm_model, lgbm_params2, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid2.best_params_
# Out[793]:
# {'Num_leaves': 60,
#  'colsample_bytree': 0.9,
#  'learning_rate': 0.18,
#  'max_depth': 3,
#  'n_estimators': 650}
lgbm_final2 = lgbm_model.set_params(**lgbm_best_grid2.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final2, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse
# RMSE: 0.15381460929184793(LightGBM)
# Önceki RMSE:  0.197 (LightGBM)

################################################
# XGBRegressor
################################################

xgboost_model = XGBRegressor(use_label_encoder=False, eval_metric='logloss', random_state= 18)

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}
# Out[535]:
# {'colsample_bytree': 1,
#  'learning_rate': 0.1,
#  'max_depth': 5,
#  'n_estimators': 200}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
xgboost_best_grid.best_params_

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(xgboost_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

# Önceki RMSE: 0.1837 (XGBoost)
# Out[543]: 0.1826537896966512

################################################
# CatBoost
################################################

catboost_model = CatBoostRegressor(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500 ],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]
                   }
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid.best_params_
# Out[178]: {'depth': 6, 'iterations': 500, 'learning_rate': 0.1}
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(catboost_final, X, y, cv=10, scoring="neg_mean_squared_error")))
# Out:  0.15827136442770523
# Önceki RMSE: 0.151 (CatBoost)
# Defoult RMSE degeri daha iyi!


catboost_model = CatBoostRegressor(random_state=17, verbose=False)

catboost_params2 = {"iterations": [500, 550],
                    "learning_rate": [0.095, 0.1],
                    "depth": [ 5, 6, 7, None]}


catboost_best_grid2 = GridSearchCV(catboost_model, catboost_params2, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid2.best_params_
# Out[107]: {'depth': 7, 'iterations': 550, 'learning_rate': 0.095}

catboost_final2 = catboost_model.set_params(**catboost_best_grid2.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(catboost_final2, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse
# Out[770]: 0.15261851772012042
# Önceki RMSE: 0.151 (CatBoost)

# modelin random değerleri çok daha verimli o yüzden onu kullanıyorum

################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')



plot_importance(lgbm_final, X)
plot_importance(catboost_model, X)



