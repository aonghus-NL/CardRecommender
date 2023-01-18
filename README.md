Om de gegevens te verkrijgen die benodigd zijn voor het maken van een relevante kaartsuggestie moeten de CSV-bestanden eerst worden opgeslagen en uitgelezen.

Het importeren van de CSV-bestanden:
rating_df = pd.read_csv('ratings.csv', usecols=['userId', 'cardId', 'leeftijd', 'rating'],
                        dtype={'userId': 'Int64', 'cardId': 'Int64', 'leeftijd': 'Int64', 'rating': 'int32'}, sep=';')

usecols
De verschillende kolommen die gebruikt worden in de dataset.

dtype
De data typen van de kolommen in de dataset.

sep=';'
Aanduiding van het scheidingsteken tussen de data.


cardId;title;trefwoorden;
65;Van harte Gefeliciteerd met katten;poes,poezen,gefeliciteerd,van-harte,felicitatie,kat

CardId: de id van de kaart:	65
Title: De titel van de kaart:	Gefeliciteerd met katten
Trefwoorden: de trefwoorden van de kaart:	Poes,poezen,gefeliciteerd,van-harte, felicitatie kat


userId;cardId;contact_id;leeftijd;rating
12658;249212;16758366;75;5

UserId: de id van de klant	12658
CardId: de id van de kaart	249212
Contact_id: de id van het kalendermoment		16758366
Leeftijd: de leeftijd van het kalendermoment		75
Rating: een variabele die aangepast kan worden om bepaalde kaarten vaker terug te laten komen (hoe hoger de rating hoe eerder het kaartje gesuggereerd zal worden)	5
*Er zijn weinig contacten waarvan het contact_id en leeftijd bekend van zijn.

Om uit deze CSV-bestanden data te lezen moeten de tabellen samen worden gevoegd.
df = pd.merge(rating_df, card_df, on='cardId')
De tabellen worden samen gevoegd op cardid.

<img width="454" alt="image" src="https://user-images.githubusercontent.com/55138203/213212220-38877d62-ffc4-4b6d-ac2d-b386a0a76804.png">
 
De output 


combine_card_rating = df.dropna(axis=0, subset=['title'])

Combine_card_rating: wanneer er een missende waarde van titel in de samengevoegde tabel is dan wordt deze rij vervolgens verwijderd. Daardoor wordt voorkomen dat er kaarten worden meegenomen in de suggesties die geen titel hebben.


card_ratingCount = (combine_card_rating.
groupby(by=['title'])['rating'].
count().
reset_index().
rename(columns={'rating': 'totalRatingCount'})
[['title', 'totalRatingCount']]
)
card_ratingCount: de waardes van de ratings worden gecombineerd zodat er hierop gefilterd kan worden. Hierdoor wordt voorkomen dat er kaarten worden meegenomen in de suggestie die niet relevant zijn. Er wordt een nieuwe tabel aangemaakt bestaande uit ‘title’ en totalRatingcount.

rating_with_totalRatingCount = combine_card_rating.merge(card_ratingCount, left_on='title', right_on='title', how='left')
De nieuwe tabel met totalRatingcount wordt samen gevoegd met de al bestaande samen gevoegde tabel. De nieuwe samen gevoegde tabel krijgt de naam rating_with_totalRatingCount.

<img width="454" alt="image" src="https://user-images.githubusercontent.com/55138203/213212279-cd52ee0e-ca2c-428d-9de9-a6a2fed71fdf.png">

De output


query_index = 9
popularity_threshold = 40
Query_index: de index die opgevraagd wordt vanuit de samen gevoegde datasets. Vanuit dit kaartje zullen de kaartsuggesties komen. 

Popularity index: Door de rating die meegegeven wordt per kaartje kan er gefilterd worden 
op een minimale rating. Wanneer de drempelwaarde op bijvoorbeeld 40 wordt gezet, worden alleen de kaarten getoond die een rating hebben van 40 of meer. Hierdoor kunnen de kaarten worden gesuggereerd waarbij meerdere klanten dezelfde kaarten hebben gekocht. Stel Klant A heeft een tijgerkaartje en een olifantenkaartje gekocht. Klant B is opzoek naar een nieuw kaartje. Een eerdere aankoop van klant B was het olifantenkaartje. Het algoritme ziet het verband tussen de aankopen van Klant A en B en zal Klant B een tijgerkaartje aanraden. Door de drempelwaarde op bijvoorbeeld 40 te zetten moeten 40 klanten de kaart combinatie hebben gekocht.

 
rating_popular_card = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
Er wordt gefilterd op de kaarten met de waardes van de popularity_threshold die gelijk is of hoger is aan de eerder ingestelde drempelwaarde. Van de overblijvende waardes wordt een nieuwe tabel gemaakt die rating_popular_card genoemd wordt.

<img width="454" alt="image" src="https://user-images.githubusercontent.com/55138203/213212312-7641b99b-a8f4-463e-8bb9-347262afa3d3.png">

De output


card_features_df = rating_popular_card.pivot_table(index='title', columns='userId', values='rating').fillna(0)
De tabel wordt aangepast naar op de y-as title en op de x-as userId, de waardes die worden toegevoegd zijn met de ratings. De waardes die niet bekend zijn worden toegevoegd als 0.  De nieuwe tabel heet card_features_df.

<img width="454" alt="image" src="https://user-images.githubusercontent.com/55138203/213212362-a27d291d-6e4d-4b64-b740-d27a1c49bf72.png">


De output van card_features_df


card_features_df_matrix = csr_matrix(card_features_df.values)
De nieuwe tabel wordt omgezet naar een CSR-matrix (Compressed Sparse Row matrix) zodat hier een KNN-model van gemaakt kan worden. 

<img width="112" alt="image" src="https://user-images.githubusercontent.com/55138203/213213053-613f4999-6e9e-4adc-923b-88187b46d4d9.png">

 
De output van card_features_df_matrix met als kolommen cardId, userId en de bijbehorende rating. 



<img width="674" alt="image" src="https://user-images.githubusercontent.com/55138203/213212703-e1b7f2b7-44e8-4129-8f5e-bde52bb2285a.png">



Nearest Neighbour algoritme:
 
Figuur 1
Een korte uitleg over het KNN-algortime. Voor een uitgebreidere uitleg zie het algoritme onderzoek.

KNN is een classificatie algoritme dit wil zeggen dat onbekende data punten toegewezen worden aan al bestaande klassen met bekende data punten. Er wordt gekeken naar de klassen van de dichtstbijzijnde buren. Wanneer er meer buren met de klasse blue (figuur 1) zijn dan buren met de klasse orange (figuur 1)  dan zal het nieuwe datapunt (waarschijnlijk) ook van de klasse blue zijn en wordt deze toegewezen aan deze klasse. 

Hoe werkt dit in een kaartsuggestie algoritme? Er wordt gekeken naar eerdere aankopen van klanten. Wanneer klant A opzoek is naar een kaartje om naar de jarige te sturen worden de laatste aankoop voor de jarige gezien als onbekend datapunt. Vervolgens wordt er gekeken naar alle andere klanten die hetzelfde kaartje hebben gekocht als klant A. Dit zijn de bekende data punten. Hoe vaker dezelfde kaarten zijn gekocht door dezelfde klanten hoe groter de kans dat deze kaarten zullen getoond worden als suggestie voor klant A.  



 

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(card_features_df_matrix)
Om de afstand te kunnen berekenen tussen 2 buren wordt hier een matrix van gemaakt.
Vanuit dit model kan de afstand worden berekend tussen 2 punten. 
Het berekenen tussen de afstand van 2 punten wordt gedaan door cosine similairty
metric='cosine'
Hierdoor wordt er gezocht naar de dichtstbijzijnde buren van het datapunt. Er wordt in een 3-dimensionale ruimte alle cosinussen berekend. Hoe kleiner de cosinus hoe dichterbij het datapunt. Deze cosinus worden stuk voor stuk met elkaar vergeleken dit wordt gedaan door:
algorithm='brute'

Vervolgens worden de kleinste afstanden met de daarbij behorende cardId toegewezen aan 2 variabelen. Daar worden ook het gewenste aantal buren aangegeven
distances, indices = model_knn.kneighbors(card_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=5)
 
Daarna worden de berekende buren oftewel de kaartsuggesties getoond.
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(card_features_df.index[query_index]))

    else:
        print('{0}: {1}, with distance of {2}'.format(i, card_features_df.index[indices.flatten()[i]],
                                                       distances.flatten()[i]))

