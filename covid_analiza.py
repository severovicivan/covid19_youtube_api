# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:27:18 2020

@author: ivan.severovic
"""
import re
import glob
import demoji
import text_hr
import numpy as np
import pandas as pd
import collections
import networkx as nx
from datetime import datetime, timedelta
from textblob import TextBlob
from langdetect import detect
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib import dates as mpl_dates


def concat_dataframes():
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    #combine all files in the list
    combined = pd.concat([pd.read_csv(f) for f in all_filenames if f != 'ListaVidea.csv'])
    combined['Keyword'] = 'korona cjepivo'
    #export to csv
    # combined.to_csv( "ListaVidea.csv", index=False, encoding='utf-8-sig')
    combined.to_csv('ListaVidea.csv',index=False, mode='a', encoding='utf-8-sig', header=False)
    return combined
    
def micanje_emotikona():
    podaci = pd.read_csv('HrvatskiKomentari.csv')
    demoji.download_codes()
    podaci['Comment'] = podaci['Comment'].apply(lambda x: demoji.replace(x,""))
    podaci['Comment'].replace('', np.nan, inplace=True)
    podaci.dropna(subset=['Comment'], inplace=True)
    podaci.to_csv("CistaListaVidea.csv", index=False, encoding='utf-8-sig')
    return podaci

def detekcija_jezika():
    podaci = pd.read_csv('JedinstvenaListaVidea.csv')
    podaci['Language'] = 0
    brojac = 0
    komentari = podaci['Comment']
    for i in range (0,len(komentari)):
        temp = komentari.iloc[i]
        brojac += 1
        try:
            podaci['Language'].iloc[i] = detect(temp)
        except:
            podaci['Language'].iloc[i] = 'error'
    print(podaci[podaci['Language']=='hr']['Language'].value_counts())
    podaci_hrvatski = podaci[podaci['Language']=='hr']
    podaci_hrvatski.to_csv('HrvatskiKomentari.csv',index = False, encoding='utf-8-sig')
    return podaci_hrvatski

def frekvencije_rijeci():
    podaci = pd.read_csv('HrvatskiKomentari.csv')
    # stem, suffix  = ".+(e|a|i|u)va juci|smo|ste|jmo|jte|ju|la|le|li|lo|mo|na|ne|ni|no|te|ti|se|hu|h|j|m|n|o|t|v|s| ".strip().split(' ')
    # rule = re.compile(r'^('+stem+')('+suffix+r')$')
    komentari = podaci['Comment'].tolist()
    videozapisi = podaci['Video Title'].tolist()
    
    stopwords = []
    for word_base in text_hr.get_all_std_words():
        stopwords.append(word_base[0].replace('š', 's').replace('Š', 'S').replace('đ', 'd').replace('Đ', 'D').replace('dž','d').replace('DŽ','D').replace('č', 'c').replace('Č', 'C').replace('ć', 'c').replace('Ć', 'C').replace('ž', 'z').replace('Ž', 'Z'))
    
    lista_rijeci = []
    for komentar in komentari:
        razdvojeni_komentar = komentar.split()
        for rijec in razdvojeni_komentar:
            if rijec.lower() not in stopwords:
                rijec = re.sub(r'[^a-zA-Z]', '', rijec)
                lista_rijeci.append(rijec.lower())
            
    # stem = [(rule.match(rijec)).group(1) for rijec in lista_rijeci]
            
    frekvencije = []
    for rijec in lista_rijeci:
        frekvencije.append(lista_rijeci.count(rijec))
        
    parovi = list(zip(lista_rijeci, frekvencije))
    jedinstveni_parovi = list(set(parovi))
    sortirani_parovi = sorted(jedinstveni_parovi, key=lambda tup: tup[1])
    
    for par in sortirani_parovi:
       if par[0] in ['','je','su','to','sve','sam','kad','ce','ovo','bi','si','sta',
                     'ga','ko','ste','im','mu','nas','vas','ima','nema','ovo','kaj',
                     'reci','me','nam','koja','nije']:
           sortirani_parovi.remove(par)
           print('['+par[0]+']')
       else:
           print('('+par[0]+')')
    
    df_korpus = pd.DataFrame()
    df_korpus['Naziv mjere'] = ['Broj videozapisa',
                                'Broj komentara',
                                'Broj riječi',
                                'Broj različitih riječi']
    df_korpus['Podatak'] = [len(list(set(videozapisi))),
                            len(komentari),
                            len(lista_rijeci),
                            len(sortirani_parovi)]
    df_korpus.to_csv('Korpus.csv',index = False, encoding='utf-8-sig')
    
    df = pd.DataFrame(sortirani_parovi[-7::-1][:40], columns=['Rijec', 'Frekvencija'])
    df.to_csv('FrekvencijeRijeci.csv',index = False, encoding='utf-8-sig')
    
    return sortirani_parovi[-7::-1],df_korpus

def micanje_duplikata():
    podaci = pd.read_csv('ListaVidea.csv')
    unique_df = podaci.drop_duplicates(subset=['Comment'])
    unique_df.to_csv("JedinstvenaListaVidea.csv", index=False, encoding='utf-8-sig')
    return unique_df

def popularna_videa():
    podaci = pd.read_csv('HrvatskiKomentari.csv')
    podaci = podaci.drop_duplicates(subset=['Video ID'])
    datumi = []
    for datum in podaci['Video Published'].tolist():
        datumi.append(datum.split('T')[0] + ' ' + datum.split('T')[1][:-1])
    podaci['Video Published'] = datumi
    podaci = podaci[['Channel','Video Title','Video Published','Video Views',
                   'Video Comments','Video Likes','Video Dislikes',
                   'Video Description','Video ID']]
    most_views = podaci.sort_values('Video Views',ascending=False)
    podaci = podaci[['Channel','Video Title','Video Published','Video Likes',
                   'Video Comments','Video Views','Video Dislikes',
                   'Video Description','Video ID']]
    most_likes = podaci.sort_values('Video Likes',ascending=False)
    podaci = podaci[['Channel','Video Title','Video Published','Video Comments',
                   'Video Views','Video Likes','Video Dislikes',
                   'Video Description','Video ID']]
    most_comments = podaci.sort_values('Video Comments',ascending=False)
    return most_views, most_likes, most_comments

def lista_bridova():
    podaci = pd.read_csv('HrvatskiKomentari.csv')
    cvorovi = [tuple(x) for x in podaci[['Channel','Author']].to_numpy()]
    graph_df = pd.DataFrame()
    cvorovi1 = []
    cvorovi2 = []
    tezine = []
    for brid in cvorovi:
        obrnuti = "('"+ brid[1] +"', '"+ brid[0] +"')"
        if obrnuti in cvorovi:
            print(obrnuti)
        if(brid[0] == brid[1]):
            cvorovi.remove(brid)
        else:
            cvorovi1.append(brid[0])
            cvorovi2.append(brid[1])
            tezine.append(cvorovi.count(brid))
    graph_df['Source'] = cvorovi1
    graph_df['Target'] = cvorovi2
    graph_df['Weight'] = tezine
    graph_df = graph_df.drop_duplicates()
    graph_df.to_csv("Bridovi2.csv", index=False, encoding='utf-8-sig')
    return graph_df
    
def analiza_mreze():
    bridovi = pd.read_csv('Bridovi2.csv')
    G = nx.Graph()
    podaci = [tuple(x) for x in bridovi[['Source','Target','Weight']].to_numpy()]
    for c1,c2,tezina in podaci:
        G.add_edge(c1,c2,weight=tezina)
    autori = bridovi['Source'].tolist()
    autor = []
    stupanj = []
    stupnjevi = pd.DataFrame()
    komentari_cvorova = 0
    for cvor in G.degree(weight='weight'):
        if cvor[0] in autori:
            stupanj.append(cvor[1])
            autor.append(cvor[0])
            komentari_cvorova += int(cvor[1])
    stupnjevi['Autor'] = autor
    stupnjevi['Stupanj'] = stupanj
    najkomentiraniji = stupnjevi.sort_values('Stupanj',ascending=False)
    najkomentiraniji[:10].to_csv("KreatoriYTNet2.csv", index=False, encoding='utf-8-sig')
    
    prosjek_komentara = komentari_cvorova/len(set(autori))
    
    globalne_mjere = pd.DataFrame()
    print('\nBroj čvorova: ' + str(len(G.nodes())))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Broj čvorova',
                'Vrijednost': len(G.nodes())}, ignore_index=True)
    
    print('Broj bridova: ' + str(len(G.edges())))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Broj bridova',
                'Vrijednost': len(G.edges())}, ignore_index=True)
    
        # Prosječan broj veza
    veze_cvorova = 0
    for cvor in G.degree():
        veze_cvorova += cvor[1]
    prosjek_veza = veze_cvorova/len(G.nodes())
    print('\nProsječni stupanj grafa G je: ' + str(prosjek_veza))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Prosječni stupanj',
                    'Vrijednost': prosjek_veza}, ignore_index=True)
    
    # 3. Ukoliko je mreža težinska dodatno računati prosječnu snagu
    snage_cvorova = []
    for cvor in G.degree(weight='weight'):
        snage_cvorova.append(cvor[1])
    prosjek_snage = sum(snage_cvorova)/len(G.nodes())
    print('\nProsječna snaga grafa G2 je: ' + str(prosjek_snage) + '\n')
    globalne_mjere = globalne_mjere.append({'Mjera': 'Prosječna snaga',
                    'Vrijednost': prosjek_snage}, ignore_index=True)
    
        # 4. Odrediti broj komponenti i veličinu najveće komponente (broj čvovova i veza)
    print('\nBroj povezanih komponenti: ' + str(nx.number_connected_components(G)))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Broj povezanih komponenti',
                    'Vrijednost': nx.number_connected_components(G)}, ignore_index=True)
    najveca = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
    print('\nBroj čvorova najveće povezane komponente: ' + str(len(najveca.nodes())))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Broj čvorova najveće povezane komponente',
                    'Vrijednost': len(najveca.nodes())}, ignore_index=True)
    print('\nBroj bridova najveće povezane komponente: ' + str(len(najveca.edges())))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Broj bridova najveće povezane komponente',
                    'Vrijednost': len(najveca.edges())}, ignore_index=True)
    
        # 5. Odrediti mjere udaljenosti za cijelu mrežu (avg. shortest path length, diameter, eccentricity)
    putevi = []
    for c in nx.connected_components(G):
        putevi.append(nx.average_shortest_path_length(G.subgraph(c),weight='weight'))
    print('\nProsječna duljina najkraćih puteva: ' + str(sum(putevi)/len(putevi)))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Prosječna duljina najkraćih puteva',
                    'Vrijednost': sum(putevi)/len(putevi)}, ignore_index=True)
    
    dijametri = []
    for c in nx.connected_components(G):
        dijametri.append(nx.diameter(G.subgraph(c)))
    print('\nDijametar mreže: ' + str(sum(dijametri)/len(dijametri)))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Dijametar mreže',
                    'Vrijednost': sum(dijametri)/len(dijametri)}, ignore_index=True)
    
    ekscentricnosti = []
    for c in nx.connected_components(G):
        ekscentricnost = nx.eccentricity(G.subgraph(c)).values()
        ekscentricnosti.append(sum(ekscentricnost)/len(ekscentricnost))
    print('\nEkscentričnost mreže: ' + str(sum(ekscentricnosti)/len(ekscentricnosti)))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Ekscentričnost mreže',
                    'Vrijednost': sum(ekscentricnosti)/len(ekscentricnosti)}, ignore_index=True)
    
    # 6. Izračunati globalnu učinkovitost
    print('\nGlobalna učinkovitost grafa: ' + str(nx.global_efficiency(G)))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Globalna učinkovitost',
                    'Vrijednost': nx.global_efficiency(G)}, ignore_index=True)
    
    # 7. Odrediti globalni koeficijent grupiranja - broj trokuta/svi mogući
    print('\nGlobalni koeficijent grupiranja: ' + str(nx.transitivity(G)))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Globalni koeficijent grupiranja',
                    'Vrijednost': nx.transitivity(G)}, ignore_index=True)
    
    # 8. Odrediti prosječni koeficijent grupiranja 
    print('\nProsječni koeficjent grupiranja: ' + str(nx.average_clustering(G,weight='weight')))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Prosječni koeficjent grupiranja',
                    'Vrijednost': nx.average_clustering(G,weight='weight')}, ignore_index=True)
    
    # 9. Izračunati asortativnost obzirom na stupanj čvora - hubovi se baš ne spajaju međusobno
    print('\nAsortativnost: ' + str(nx.degree_assortativity_coefficient(G,weight='weight')))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Asortativnost',
                    'Vrijednost': nx.degree_assortativity_coefficient(G,weight='weight')}, ignore_index=True)
    
    #     # 10. Nacrtati dijagram disturibucije stupnjeva - POLINOMNA
    # stupnjevi = [G.degree(n) for n in G.nodes()]
    # plt.figure(figsize=(11, 5))
    # plt.hist(stupnjevi)
    # plt.xticks(np.arange(min(stupnjevi), max(stupnjevi)+1, 1.0))
    # plt.xlabel('Stupanj čvora', fontsize=18)
    # plt.ylabel('Broj čvorova', fontsize=18)
    # plt.title('Dijagram distribucije stupnjeva',fontsize=18)
    # plt.show()
    
    # # 10.5 Nacrtati dijagram disturibucije snage
    # plt.figure(figsize=(11, 5))
    # plt.hist(snage_cvorova)
    # plt.xticks(np.arange(min(snage_cvorova), max(snage_cvorova)+1, 5))
    # plt.xlabel('Snaga čvora', fontsize=18)
    # plt.ylabel('Broj čvorova', fontsize=18)
    # plt.title('Dijagram distribucije snage',fontsize=18)
    # plt.show()
    # print('Broj najviše komentara: ' + str(G.degree('IvanPernarTV',weight='weight')))
    
   
    print('Broj nepostojećih bridova: ' + str(len(list(nx.non_edges(G)))))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Broj nepostojećih bridova',
                    'Vrijednost': len(list(nx.non_edges(G)))}, ignore_index=True)
    print('Gustoća mreže: '+ str(nx.density(G)))
    globalne_mjere = globalne_mjere.append({'Mjera': 'Gustoća mreže',
                    'Vrijednost': nx.density(G)}, ignore_index=True)
    print('Jedinstvenih autora: ' + str(len(set(autori))))
    # globalne_mjere = globalne_mjere.append({'Mjera': 'Broj kreatora videozapisa',
    #                 'Vrijednost': len(set(autori))}, ignore_index=True)   
    
    
    najaktivniji = pd.DataFrame()
    korisnici = []
    komentari = []
    for cvor in G.degree(weight='weight'):
        if cvor[0] not in autori:
            korisnici.append(cvor[0])
            komentari.append(cvor[1])
    najaktivniji['Korisnik'] = korisnici
    najaktivniji['Broj komentara'] = komentari
    najaktivniji = najaktivniji.sort_values('Broj komentara',ascending=False)
    najaktivniji[:10].to_csv("KomentatoriYTNet2.csv", index=False, encoding='utf-8-sig')
    
    # 12. Odrediti prosječnu centralnost blizine
    blizine_cvorova = []
    for n, cc in nx.closeness_centrality(G).items():
        blizine_cvorova.append(cc)
    prosjek_blizine = sum(blizine_cvorova)/len(G.nodes())
    print('\nProsječna centralnost blizine grafa G2 je: ' + str(prosjek_blizine) + '\n')
    globalne_mjere = globalne_mjere.append({'Mjera': 'Prosječna centralnost blizine',
                    'Vrijednost': prosjek_blizine}, ignore_index=True)
    
    # 13. Odrediti prosječnu međupoloženost
    medupolozenosti_cvorova = []
    for n, bc in nx.betweenness_centrality(G,weight='weight').items():
        medupolozenosti_cvorova.append(bc)
    prosjek_medupolozenosti = sum(medupolozenosti_cvorova)/len(G.nodes())
    print('Prosječna centralnost međupoloženosti grafa G2 je: ' + str(prosjek_medupolozenosti) + '\n')
    globalne_mjere = globalne_mjere.append({'Mjera': 'Prosječna centralnost međupoloženosti',
                    'Vrijednost': prosjek_medupolozenosti}, ignore_index=True)

    
    print('Broj komentatora: ' + str(len(set(korisnici))))
    # globalne_mjere = globalne_mjere.append({'Mjera': 'Broj komentatora videozapisa',
    #                 'Vrijednost': len(set(korisnici))}, ignore_index=True)
    # print('Prosječan broj komentara: ' + str(prosjek_komentara))
    # globalne_mjere = globalne_mjere.append({'Mjera': 'Prosječan broj komentara',
    #                 'Vrijednost': prosjek_komentara}, ignore_index=True)
    
    globalne_mjere.to_csv("GlobalneMjereYTNet2.csv", index=False, encoding='utf-8-sig')
    
    #               Analiza mreže na lokalnoj razini
    # 11. Odrediti centralne čvorove prema različitim mjerama centralnosti
    # 11.a) Centralnost stupnja čvora
    # Dohvaćanje 10 najvećih jedinstvenih vrijednosti centralnosti stupnja čvora: top_dcs
    top_dcs = sorted(set(nx.degree_centrality(G).values()), reverse=True)[0:10]
    # Kreiranje liste čvorova koji imaju 10 najvećih vrijednosti za centralnost stupnja čvora(degree centrality)
    top_connected = []
    for n, dc in nx.degree_centrality(G).items():
        if dc in top_dcs:
            top_connected.append((n,dc))     
    # Čvorovi s najvećim centralnostima stupnja čvora
    print('\n10 osoba s najvećom centralnosti stupnja čvora:\n' + str(top_connected[0:10]))
    top_connected.sort(key=lambda tup: tup[1])
    max_s = pd.DataFrame(reversed(top_connected[-10:]), columns =['Autor', 'Centralnost stupnja'])
    max_s.to_csv("CentStupnjaYTNet2.csv", index=False, encoding='utf-8-sig')
    
    # 11.b) Centralnost međupoloženosti
    # Dohvaćanje 10 najvećih jedinstvenih vrijednosti centralnosti međupoloženosti: top_bcs
    top_bcs = sorted(set(nx.betweenness_centrality(G,weight='weight').values()), reverse=True)[0:10]
    # Kreiranje liste čvorova koji imaju 10 najvećih vrijednosti za centralnost stupnja čvora(degree centrality)
    najbitniji = []
    for n, bc in nx.betweenness_centrality(G,weight='weight').items():
        if bc in top_bcs:
            najbitniji.append((n,bc))     
    # Čvorovi s najvećim centralnostima međupoloženosti
    print('\n10 osoba s najvećom centralnosti međupoloženosti:\n' + str(najbitniji[0:10]))
    najbitniji.sort(key=lambda tup: tup[1])
    max_m = pd.DataFrame(reversed(najbitniji[-10:]), columns =['Autor', 'C. međupoloženosti'])
    max_m.to_csv("CentMedYTNet2.csv", index=False, encoding='utf-8-sig')
    
    # 11.c) Centralnost blizine
    # Dohvaćanje 10 najvećih jedinstvenih vrijednosti centralnosti međupoloženosti: top_bcs
    top_ccs = sorted(set(nx.closeness_centrality(G).values()), reverse=True)[0:10]
    # Kreiranje liste čvorova koji imaju 10 najvećih vrijednosti za centralnost stupnja čvora(degree centrality)
    najpristupacniji = []
    for n, cc in nx.closeness_centrality(G).items():
        if cc in top_ccs:
            najpristupacniji.append((n,cc))     
    # Čvorovi s najvećim centralnostima blizine
    print('\n10 osoba s najvećom centralnosti blizine:\n' + str(najpristupacniji[0:10]))
    najpristupacniji.sort(key=lambda tup: tup[1])
    max_b = pd.DataFrame(reversed(najpristupacniji[-10:]), columns =['Autor', 'Centralnost blizine'])
    max_b.to_csv("CentBlizineYTNet2.csv", index=False, encoding='utf-8-sig')
    
    # 11.d) Centralnost svojstvenog vektora
    # Dohvaćanje 10 najvećih jedinstvenih vrijednosti centralnosti svojstvenog vektora: top_ecs
    top_ecs = sorted(set(nx.eigenvector_centrality(G,weight='weight').values()), reverse=True)[0:10]
    # Kreiranje liste čvorova koji imaju 10 najvećih vrijednosti za centralnost svekt čvora(eigenvector centrality)
    najeigen = []
    for n, cc in nx.eigenvector_centrality(G,weight='weight').items():
        if cc in top_ecs:
            najeigen.append((n,cc))     
    # Čvorovi s najvećim centralnostima blizine
    print('\n10 osoba s najvećom centralnosti blizine:\n' + str(najeigen[0:10]))
    najeigen.sort(key=lambda tup: tup[1])
    max_e = pd.DataFrame(reversed(najeigen[-10:]), columns =['Autor', 'Centralnost svojstvenog vektora'])
    max_e.to_csv("CentEigenYTNet2.csv", index=False, encoding='utf-8-sig')
    
    return max_s, max_m, max_b, max_e

def dodatno_ciscenje():
    podaci = pd.read_csv('FrekvencijeRijeci.csv')
    for index, rows in podaci.iterrows():
        if rows[0] in ['','je','su','to','sve','sam','kad','ce','ovo','bi','si','sta',
                     'ga','ko','ste','im','mu','nas','vas','ima','nema','ovo','kaj',
                     'reci','me','nam','koja','nije','smo','ove','ovom','svi','svima',
                     'svim','vam','mene','koje','se','svaka','bilo','zasto','koliko']:
            podaci = podaci.drop(index=index)
    podaci[:20].to_csv('FrekvencijeRijeciYTNet2.csv',index = False, encoding='utf-8-sig')
    return podaci

def korelacija():
    podaci = [{"SlucajeviSvijet":81154259,"SlucajeviHrvatska":205246,"UmrliSvijet":1772222,"UmrliHrvatska":3739,"IzlijeceniSvijet":57305066,"IzlijeceniHrvatska":193471,"Datum":"2020-12-28 10:57"},{"SlucajeviSvijet":80731992,"SlucajeviHrvatska":204930,"UmrliSvijet":1764913,"UmrliHrvatska":3671,"IzlijeceniSvijet":56924993,"IzlijeceniHrvatska":191226,"Datum":"2020-12-27 10:59"},{"SlucajeviSvijet":80222683,"SlucajeviHrvatska":204312,"UmrliSvijet":1757995,"UmrliHrvatska":3613,"IzlijeceniSvijet":56490614,"IzlijeceniHrvatska":189055,"Datum":"2020-12-26 11:03"},{"SlucajeviSvijet":79771523,"SlucajeviHrvatska":203962,"UmrliSvijet":1750067,"UmrliHrvatska":3548,"IzlijeceniSvijet":56160251,"IzlijeceniHrvatska":186533,"Datum":"2020-12-25 12:43"},{"SlucajeviSvijet":79086170,"SlucajeviHrvatska":202319,"UmrliSvijet":1738168,"UmrliHrvatska":3464,"IzlijeceniSvijet":55674767,"IzlijeceniHrvatska":183532,"Datum":"2020-12-24 11:22"},{"SlucajeviSvijet":78380027,"SlucajeviHrvatska":200086,"UmrliSvijet":1724394,"UmrliHrvatska":3394,"IzlijeceniSvijet":55148950,"IzlijeceniHrvatska":180735,"Datum":"2020-12-23 11:10"},{"SlucajeviSvijet":77740380,"SlucajeviHrvatska":197323,"UmrliSvijet":1709474,"UmrliHrvatska":3328,"IzlijeceniSvijet":54619686,"IzlijeceniHrvatska":178880,"Datum":"2020-12-22 11:40"},{"SlucajeviSvijet":77184964,"SlucajeviHrvatska":195728,"UmrliSvijet":1699878,"UmrliHrvatska":3257,"IzlijeceniSvijet":54101196,"IzlijeceniHrvatska":176366,"Datum":"2020-12-21 11:06"},{"SlucajeviSvijet":76687903,"SlucajeviHrvatska":194962,"UmrliSvijet":1693209,"UmrliHrvatska":3177,"IzlijeceniSvijet":53804828,"IzlijeceniHrvatska":173158,"Datum":"2020-12-20 11:36"},{"SlucajeviSvijet":76037327,"SlucajeviHrvatska":192987,"UmrliSvijet":1681629,"UmrliHrvatska":3101,"IzlijeceniSvijet":53304349,"IzlijeceniHrvatska":169768,"Datum":"2020-12-19 11:07"},{"SlucajeviSvijet":75299629,"SlucajeviHrvatska":190235,"UmrliSvijet":1668691,"UmrliHrvatska":3023,"IzlijeceniSvijet":52869567,"IzlijeceniHrvatska":165915,"Datum":"2020-12-18 11:11"},{"SlucajeviSvijet":74537459,"SlucajeviHrvatska":186963,"UmrliSvijet":1655307,"UmrliHrvatska":2955,"IzlijeceniSvijet":52374434,"IzlijeceniHrvatska":161563,"Datum":"2020-12-17 10:58"},{"SlucajeviSvijet":73821896,"SlucajeviHrvatska":183045,"UmrliSvijet":1642044,"UmrliHrvatska":2870,"IzlijeceniSvijet":51833667,"IzlijeceniHrvatska":157773,"Datum":"2020-12-16 11:06"},{"SlucajeviSvijet":73211509,"SlucajeviHrvatska":179718,"UmrliSvijet":1628442,"UmrliHrvatska":2778,"IzlijeceniSvijet":51347757,"IzlijeceniHrvatska":155079,"Datum":"2020-12-15 11:11"},{"SlucajeviSvijet":72655939,"SlucajeviHrvatska":177358,"UmrliSvijet":1619077,"UmrliHrvatska":2705,"IzlijeceniSvijet":50874955,"IzlijeceniHrvatska":151884,"Datum":"2020-12-14 11:07"},{"SlucajeviSvijet":72126453,"SlucajeviHrvatska":175886,"UmrliSvijet":1611948,"UmrliHrvatska":2640,"IzlijeceniSvijet":50506678,"IzlijeceniHrvatska":148211,"Datum":"2020-12-13 11:05"},{"SlucajeviSvijet":71462822,"SlucajeviHrvatska":172523,"UmrliSvijet":1601628,"UmrliHrvatska":2562,"IzlijeceniSvijet":49662066,"IzlijeceniHrvatska":144691,"Datum":"2020-12-12 11:04"},{"SlucajeviSvijet":70745895,"SlucajeviHrvatska":168388,"UmrliSvijet":1588911,"UmrliHrvatska":2484,"IzlijeceniSvijet":49172210,"IzlijeceniHrvatska":140898,"Datum":"2020-12-11 11:00"},{"SlucajeviSvijet":69262556,"SlucajeviHrvatska":163992,"UmrliSvijet":1576186,"UmrliHrvatska":2420,"IzlijeceniSvijet":48019269,"IzlijeceniHrvatska":136721,"Datum":"2020-12-10 11:33"},{"SlucajeviSvijet":68587502,"SlucajeviHrvatska":159372,"UmrliSvijet":1563487,"UmrliHrvatska":2367,"IzlijeceniSvijet":47482977,"IzlijeceniHrvatska":133255,"Datum":"2020-12-09 11:27"},{"SlucajeviSvijet":67961285,"SlucajeviHrvatska":154852,"UmrliSvijet":1550701,"UmrliHrvatska":2298,"IzlijeceniSvijet":47050607,"IzlijeceniHrvatska":130869,"Datum":"2020-12-08 12:00"},{"SlucajeviSvijet":67405131,"SlucajeviHrvatska":152239,"UmrliSvijet":1541951,"UmrliHrvatska":2233,"IzlijeceniSvijet":46594335,"IzlijeceniHrvatska":127882,"Datum":"2020-12-07 11:27"},{"SlucajeviSvijet":66882253,"SlucajeviHrvatska":150353,"UmrliSvijet":1534974,"UmrliHrvatska":2174,"IzlijeceniSvijet":46257103,"IzlijeceniHrvatska":124439,"Datum":"2020-12-06 10:56"},{"SlucajeviSvijet":66252020,"SlucajeviHrvatska":147454,"UmrliSvijet":1524768,"UmrliHrvatska":2102,"IzlijeceniSvijet":45831083,"IzlijeceniHrvatska":120857,"Datum":"2020-12-05 11:12"},{"SlucajeviSvijet":65558031,"SlucajeviHrvatska":143370,"UmrliSvijet":1512223,"UmrliHrvatska":2032,"IzlijeceniSvijet":45394765,"IzlijeceniHrvatska":117148,"Datum":"2020-12-04 11:12"},{"SlucajeviSvijet":64866214,"SlucajeviHrvatska":139415,"UmrliSvijet":1499690,"UmrliHrvatska":1964,"IzlijeceniSvijet":44962234,"IzlijeceniHrvatska":113509,"Datum":"2020-12-03 11:22"},{"SlucajeviSvijet":64214449,"SlucajeviHrvatska":134881,"UmrliSvijet":1487112,"UmrliHrvatska":1916,"IzlijeceniSvijet":44462715,"IzlijeceniHrvatska":110355,"Datum":"2020-12-02 11:03"},{"SlucajeviSvijet":63608343,"SlucajeviHrvatska":131342,"UmrliSvijet":1474219,"UmrliHrvatska":1861,"IzlijeceniSvijet":44001773,"IzlijeceniHrvatska":108231,"Datum":"2020-12-01 11:21"},{"SlucajeviSvijet":63087142,"SlucajeviHrvatska":128442,"UmrliSvijet":1465368,"UmrliHrvatska":1786,"IzlijeceniSvijet":43557257,"IzlijeceniHrvatska":105199,"Datum":"2020-11-30 12:07"},{"SlucajeviSvijet":62592000,"SlucajeviHrvatska":126612,"UmrliSvijet":1458485,"UmrliHrvatska":1712,"IzlijeceniSvijet":43203107,"IzlijeceniHrvatska":101838,"Datum":"2020-11-29 10:49"},{"SlucajeviSvijet":62037905,"SlucajeviHrvatska":123693,"UmrliSvijet":1449895,"UmrliHrvatska":1655,"IzlijeceniSvijet":42831206,"IzlijeceniHrvatska":98465,"Datum":"2020-11-28 11:02"},{"SlucajeviSvijet":61331706,"SlucajeviHrvatska":119706,"UmrliSvijet":1438096,"UmrliHrvatska":1600,"IzlijeceniSvijet":42411308,"IzlijeceniHrvatska":95698,"Datum":"2020-11-27 11:05"},{"SlucajeviSvijet":60744487,"SlucajeviHrvatska":115626,"UmrliSvijet":1427188,"UmrliHrvatska":1552,"IzlijeceniSvijet":42050100,"IzlijeceniHrvatska":92349,"Datum":"2020-11-26 10:56"},{"SlucajeviSvijet":60126931,"SlucajeviHrvatska":111617,"UmrliSvijet":1415239,"UmrliHrvatska":1501,"IzlijeceniSvijet":41570679,"IzlijeceniHrvatska":89425,"Datum":"2020-11-25 11:10"},{"SlucajeviSvijet":59533128,"SlucajeviHrvatska":108014,"UmrliSvijet":1402312,"UmrliHrvatska":1445,"IzlijeceniSvijet":41173110,"IzlijeceniHrvatska":87408,"Datum":"2020-11-24 11:52"},{"SlucajeviSvijet":59002157,"SlucajeviHrvatska":105691,"UmrliSvijet":1393879,"UmrliHrvatska":1398,"IzlijeceniSvijet":40776358,"IzlijeceniHrvatska":85018,"Datum":"2020-11-23 10:56"},{"SlucajeviSvijet":58512319,"SlucajeviHrvatska":103718,"UmrliSvijet":1386778,"UmrliHrvatska":1353,"IzlijeceniSvijet":40477677,"IzlijeceniHrvatska":82380,"Datum":"2020-11-22 11:21"},{"SlucajeviSvijet":57915601,"SlucajeviHrvatska":100410,"UmrliSvijet":1377826,"UmrliHrvatska":1304,"IzlijeceniSvijet":40114821,"IzlijeceniHrvatska":80027,"Datum":"2020-11-21 10:55"},{"SlucajeviSvijet":57261775,"SlucajeviHrvatska":96837,"UmrliSvijet":1366019,"UmrliHrvatska":1257,"IzlijeceniSvijet":39749228,"IzlijeceniHrvatska":77387,"Datum":"2020-11-20 11:10"},{"SlucajeviSvijet":56583049,"SlucajeviHrvatska":93879,"UmrliSvijet":1355147,"UmrliHrvatska":1200,"IzlijeceniSvijet":39370986,"IzlijeceniHrvatska":74865,"Datum":"2020-11-19 11:00"},{"SlucajeviSvijet":55961152,"SlucajeviHrvatska":90715,"UmrliSvijet":1343709,"UmrliHrvatska":1151,"IzlijeceniSvijet":38976150,"IzlijeceniHrvatska":72673,"Datum":"2020-11-18 12:09"},{"SlucajeviSvijet":55366732,"SlucajeviHrvatska":87464,"UmrliSvijet":1332565,"UmrliHrvatska":1113,"IzlijeceniSvijet":38507343,"IzlijeceniHrvatska":70980,"Datum":"2020-11-17 10:32"},{"SlucajeviSvijet":54832578,"SlucajeviHrvatska":85519,"UmrliSvijet":1324689,"UmrliHrvatska":1082,"IzlijeceniSvijet":38147815,"IzlijeceniHrvatska":68738,"Datum":"2020-11-16 10:51"},{"SlucajeviSvijet":54344494,"SlucajeviHrvatska":84206,"UmrliSvijet":1318452,"UmrliHrvatska":1049,"IzlijeceniSvijet":37878300,"IzlijeceniHrvatska":66231,"Datum":"2020-11-15 11:22"},{"SlucajeviSvijet":53766702,"SlucajeviHrvatska":81844,"UmrliSvijet":1309703,"UmrliHrvatska":1006,"IzlijeceniSvijet":37535282,"IzlijeceniHrvatska":63748,"Datum":"2020-11-14 10:50"},{"SlucajeviSvijet":53109750,"SlucajeviHrvatska":78978,"UmrliSvijet":1299651,"UmrliHrvatska":968,"IzlijeceniSvijet":37224907,"IzlijeceniHrvatska":61264,"Datum":"2020-11-13 10:58"},{"SlucajeviSvijet":52457990,"SlucajeviHrvatska":75922,"UmrliSvijet":1290026,"UmrliHrvatska":925,"IzlijeceniSvijet":36687958,"IzlijeceniHrvatska":58649,"Datum":"2020-11-12 10:37"},{"SlucajeviSvijet":51835949,"SlucajeviHrvatska":72840,"UmrliSvijet":1279963,"UmrliHrvatska":893,"IzlijeceniSvijet":36407888,"IzlijeceniHrvatska":56434,"Datum":"2020-11-11 11:05"},{"SlucajeviSvijet":51259771,"SlucajeviHrvatska":70243,"UmrliSvijet":1269571,"UmrliHrvatska":865,"IzlijeceniSvijet":36066124,"IzlijeceniHrvatska":54854,"Datum":"2020-11-10 11:06"},{"SlucajeviSvijet":50743485,"SlucajeviHrvatska":68776,"UmrliSvijet":1262192,"UmrliHrvatska":832,"IzlijeceniSvijet":35799384,"IzlijeceniHrvatska":53002,"Datum":"2020-11-09 11:02"},{"SlucajeviSvijet":50278660,"SlucajeviHrvatska":67247,"UmrliSvijet":1256558,"UmrliHrvatska":794,"IzlijeceniSvijet":35556538,"IzlijeceniHrvatska":50775,"Datum":"2020-11-08 10:37"},{"SlucajeviSvijet":49685311,"SlucajeviHrvatska":64704,"UmrliSvijet":1249030,"UmrliHrvatska":752,"IzlijeceniSvijet":35268937,"IzlijeceniHrvatska":48410,"Datum":"2020-11-07 10:46"},{"SlucajeviSvijet":49056664,"SlucajeviHrvatska":62305,"UmrliSvijet":1239991,"UmrliHrvatska":717,"IzlijeceniSvijet":35001624,"IzlijeceniHrvatska":46021,"Datum":"2020-11-06 10:09"},{"SlucajeviSvijet":48454224,"SlucajeviHrvatska":59415,"UmrliSvijet":1231281,"UmrliHrvatska":683,"IzlijeceniSvijet":34687623,"IzlijeceniHrvatska":43376,"Datum":"2020-11-05 10:24"},{"SlucajeviSvijet":47869340,"SlucajeviHrvatska":56567,"UmrliSvijet":1220803,"UmrliHrvatska":654,"IzlijeceniSvijet":34369995,"IzlijeceniHrvatska":41070,"Datum":"2020-11-04 11:35"},{"SlucajeviSvijet":47339417,"SlucajeviHrvatska":54087,"UmrliSvijet":1211628,"UmrliHrvatska":628,"IzlijeceniSvijet":34039293,"IzlijeceniHrvatska":39380,"Datum":"2020-11-03 10:25"},{"SlucajeviSvijet":46834497,"SlucajeviHrvatska":52660,"UmrliSvijet":1205432,"UmrliHrvatska":594,"IzlijeceniSvijet":33762216,"IzlijeceniHrvatska":37332,"Datum":"2020-11-02 10:30"},{"SlucajeviSvijet":46425070,"SlucajeviHrvatska":51495,"UmrliSvijet":1200810,"UmrliHrvatska":562,"IzlijeceniSvijet":33503608,"IzlijeceniHrvatska":35039,"Datum":"2020-11-01 10:08"},{"SlucajeviSvijet":45932232,"SlucajeviHrvatska":49316,"UmrliSvijet":1194089,"UmrliHrvatska":546,"IzlijeceniSvijet":33259068,"IzlijeceniHrvatska":32818,"Datum":"2020-10-31 10:21"},{"SlucajeviSvijet":45382151,"SlucajeviHrvatska":46547,"UmrliSvijet":1187029,"UmrliHrvatska":531,"IzlijeceniSvijet":33018697,"IzlijeceniHrvatska":30910,"Datum":"2020-10-30 10:36"},{"SlucajeviSvijet":44789859,"SlucajeviHrvatska":43775,"UmrliSvijet":1179466,"UmrliHrvatska":511,"IzlijeceniSvijet":32741484,"IzlijeceniHrvatska":29233,"Datum":"2020-10-29 10:14"},{"SlucajeviSvijet":44283038,"SlucajeviHrvatska":40999,"UmrliSvijet":1172075,"UmrliHrvatska":493,"IzlijeceniSvijet":32466672,"IzlijeceniHrvatska":27770,"Datum":"2020-10-28 10:23"},{"SlucajeviSvijet":43817194,"SlucajeviHrvatska":38621,"UmrliSvijet":1165105,"UmrliHrvatska":470,"IzlijeceniSvijet":32200294,"IzlijeceniHrvatska":26840,"Datum":"2020-10-27 11:49"},{"SlucajeviSvijet":43355163,"SlucajeviHrvatska":37208,"UmrliSvijet":1159200,"UmrliHrvatska":452,"IzlijeceniSvijet":31907861,"IzlijeceniHrvatska":25837,"Datum":"2020-10-26 10:54"},{"SlucajeviSvijet":42973486,"SlucajeviHrvatska":36380,"UmrliSvijet":1155224,"UmrliHrvatska":437,"IzlijeceniSvijet":31683279,"IzlijeceniHrvatska":24799,"Datum":"2020-10-25 10:20"},{"SlucajeviSvijet":42497462,"SlucajeviHrvatska":33959,"UmrliSvijet":1149371,"UmrliHrvatska":429,"IzlijeceniSvijet":31429851,"IzlijeceniHrvatska":23785,"Datum":"2020-10-24 10:30"},{"SlucajeviSvijet":42003060,"SlucajeviHrvatska":31717,"UmrliSvijet":1142874,"UmrliHrvatska":413,"IzlijeceniSvijet":31190947,"IzlijeceniHrvatska":22910,"Datum":"2020-10-23 10:13"},{"SlucajeviSvijet":41494389,"SlucajeviHrvatska":29850,"UmrliSvijet":1136462,"UmrliHrvatska":406,"IzlijeceniSvijet":30916843,"IzlijeceniHrvatska":22064,"Datum":"2020-10-22 10:30"},{"SlucajeviSvijet":41050369,"SlucajeviHrvatska":28287,"UmrliSvijet":1129741,"UmrliHrvatska":393,"IzlijeceniSvijet":30632287,"IzlijeceniHrvatska":21435,"Datum":"2020-10-21 10:01"},{"SlucajeviSvijet":40657780,"SlucajeviHrvatska":26863,"UmrliSvijet":1123127,"UmrliHrvatska":382,"IzlijeceniSvijet":30361708,"IzlijeceniHrvatska":20962,"Datum":"2020-10-20 10:03"},{"SlucajeviSvijet":40289836,"SlucajeviHrvatska":25973,"UmrliSvijet":1118443,"UmrliHrvatska":374,"IzlijeceniSvijet":30122268,"IzlijeceniHrvatska":20529,"Datum":"2020-10-19 10:10"},{"SlucajeviSvijet":39979677,"SlucajeviHrvatska":25580,"UmrliSvijet":1114886,"UmrliHrvatska":363,"IzlijeceniSvijet":29897007,"IzlijeceniHrvatska":20053,"Datum":"2020-10-18 10:41"},{"SlucajeviSvijet":39593885,"SlucajeviHrvatska":24761,"UmrliSvijet":1109249,"UmrliHrvatska":355,"IzlijeceniSvijet":29661122,"IzlijeceniHrvatska":19562,"Datum":"2020-10-17 10:33"},{"SlucajeviSvijet":39182095,"SlucajeviHrvatska":23665,"UmrliSvijet":1103056,"UmrliHrvatska":345,"IzlijeceniSvijet":29381358,"IzlijeceniHrvatska":19087,"Datum":"2020-10-16 10:24"},{"SlucajeviSvijet":38752973,"SlucajeviHrvatska":22534,"UmrliSvijet":1096962,"UmrliHrvatska":344,"IzlijeceniSvijet":29129637,"IzlijeceniHrvatska":18628,"Datum":"2020-10-15 10:00"},{"SlucajeviSvijet":38370434,"SlucajeviHrvatska":21741,"UmrliSvijet":1090921,"UmrliHrvatska":334,"IzlijeceniSvijet":28853981,"IzlijeceniHrvatska":18197,"Datum":"2020-10-14 10:00"},{"SlucajeviSvijet":38049049,"SlucajeviHrvatska":20993,"UmrliSvijet":1085482,"UmrliHrvatska":330,"IzlijeceniSvijet":28608404,"IzlijeceniHrvatska":17889,"Datum":"2020-10-13 10:05"},{"SlucajeviSvijet":37755013,"SlucajeviHrvatska":20621,"UmrliSvijet":1081508,"UmrliHrvatska":327,"IzlijeceniSvijet":28361454,"IzlijeceniHrvatska":17582,"Datum":"2020-10-12 10:05"},{"SlucajeviSvijet":37475839,"SlucajeviHrvatska":20440,"UmrliSvijet":1077594,"UmrliHrvatska":324,"IzlijeceniSvijet":28117060,"IzlijeceniHrvatska":17298,"Datum":"2020-10-11 10:01"},{"SlucajeviSvijet":37121450,"SlucajeviHrvatska":19932,"UmrliSvijet":1072852,"UmrliHrvatska":317,"IzlijeceniSvijet":27903233,"IzlijeceniHrvatska":16953,"Datum":"2020-10-10 10:00"},{"SlucajeviSvijet":36761330,"SlucajeviHrvatska":19446,"UmrliSvijet":1066952,"UmrliHrvatska":313,"IzlijeceniSvijet":27673859,"IzlijeceniHrvatska":16695,"Datum":"2020-10-09 10:01"},{"SlucajeviSvijet":36402301,"SlucajeviHrvatska":18989,"UmrliSvijet":1060576,"UmrliHrvatska":310,"IzlijeceniSvijet":27418740,"IzlijeceniHrvatska":16473,"Datum":"2020-10-08 10:01"},{"SlucajeviSvijet":36053143,"SlucajeviHrvatska":18447,"UmrliSvijet":1054716,"UmrliHrvatska":309,"IzlijeceniSvijet":27156099,"IzlijeceniHrvatska":16308,"Datum":"2020-10-07 10:00"},{"SlucajeviSvijet":35708182,"SlucajeviHrvatska":18084,"UmrliSvijet":1046049,"UmrliHrvatska":304,"IzlijeceniSvijet":26877331,"IzlijeceniHrvatska":16192,"Datum":"2020-10-06 11:38"},{"SlucajeviSvijet":35404671,"SlucajeviHrvatska":17797,"UmrliSvijet":1041862,"UmrliHrvatska":300,"IzlijeceniSvijet":26630686,"IzlijeceniHrvatska":16031,"Datum":"2020-10-05 10:01"},{"SlucajeviSvijet":35139699,"SlucajeviHrvatska":17659,"UmrliSvijet":1038021,"UmrliHrvatska":298,"IzlijeceniSvijet":26128410,"IzlijeceniHrvatska":15849,"Datum":"2020-10-04 10:01"},{"SlucajeviSvijet":34836028,"SlucajeviHrvatska":17401,"UmrliSvijet":1033330,"UmrliHrvatska":293,"IzlijeceniSvijet":25897996,"IzlijeceniHrvatska":15661,"Datum":"2020-10-03 10:02"},{"SlucajeviSvijet":34484731,"SlucajeviHrvatska":17160,"UmrliSvijet":1027661,"UmrliHrvatska":291,"IzlijeceniSvijet":25675090,"IzlijeceniHrvatska":15423,"Datum":"2020-10-02 10:00"}]
    datum = []
    slucajevi = []
    slucajevi2 = []
    for podatak in podaci:
        slucajevi.append(podatak['SlucajeviHrvatska'])
        datum.append(podatak['Datum'])
    for podatak2 in podaci[1:]:
        slucajevi2.append(podatak2['SlucajeviHrvatska'])
    slucajevi3 = [a - b for a, b in zip(slucajevi, slucajevi2)]
    print(slucajevi3)
    df = pd.DataFrame()
    datumi = []
    for d in datum:
        datumi.append(d.split(' ')[0])
    df['Datum'] = datumi[:-1]
    df['Slucajevi'] = slucajevi3
    df.to_csv('SlucajeviYTNet2.csv',index = False, encoding='utf-8-sig')
    return df

def pearsonova_korelacija():
    df_slucajevi = pd.read_csv('SlucajeviYTNet2.csv')
    df_objave = pd.read_csv('ObjaveKomentara.csv')
    spojeni = df_slucajevi.join(df_objave.set_index('Datum'), on='Datum')
    spojeni = spojeni.dropna()
    list1 = spojeni['Slucajevi']
    list2 = spojeni['Broj komentara']
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    x = spojeni['Slucajevi']
    y = spojeni['Broj komentara']
    slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=0, marker='s', label='Data points')
    ax.plot(x, intercept + slope * x, label=line)
    ax.set_xlabel('Broj zaraženih')
    ax.set_ylabel('Broj komentara')
    ax.legend(facecolor='white')
    plt.show()
    # corr, _ = pearsonr(list1, list2) 
    # print('Pearsons correlation: %.3f' % corr) 
    return spojeni

def objave_po_danima():
    podaci = pd.read_csv('HrvatskiKomentari.csv')
    naslov = podaci['Video Title'].tolist()
    komentar = podaci['Comment'].tolist()
    objava_videa = []
    objava_komentara = []
    
    for datum in podaci['Video Published'].tolist():
        objava_videa.append(datum.split('T')[0])        
    for datum in podaci['Comment Published'].tolist():
        objava_komentara.append(datum.split('T')[0])
        
    naslov_datum = list(zip(naslov,objava_videa))
    jedinstveni_videi = list(set(naslov_datum))
    
    def sort_date(record):
        return datetime.strptime(record[1], "%Y-%m-%d")
    
    def sort_date_0(record):
        return datetime.strptime(record[0], "%Y-%m-%d")
    
    print(sorted(jedinstveni_videi, key=sort_date, reverse=False))
    
    videa_sort = sorted(jedinstveni_videi, key=sort_date, reverse=False)
    datumi = [datum[1] for datum in videa_sort]
    frekvencije_datuma = []
    for datum in datumi:
        frekvencije_datuma.append(datumi.count(datum))
    frekvencije = list(zip(datumi,frekvencije_datuma))
    jedinstvene_frekvencije = list(set(frekvencije))
    
    # print(sorted(jedinstvene_frekvencije, key=sort_date_0, reverse=False))
    
    c = collections.defaultdict(list)
    for a,b in videa_sort:
        c[b].extend([a])  # add to existing list or create a new one
    
    datum_naslov = list(c.items())
    
    df1 = pd.DataFrame(jedinstvene_frekvencije,columns=['Datum','Broj videa'])
    df2 = pd.DataFrame(datum_naslov,columns=['Datum','Naslov'])
    df_kombinirani = pd.merge(df1, df2, on="Datum")

    df_kombinirani['Datum'] = pd.to_datetime(df_kombinirani['Datum'])
    df_kombinirani.sort_values('Datum',inplace=True)
    
    # FREKVENCIJE VIDEA - TJEDNI
    x_datumi = []
    y_frekvencije = []
    df_tjedni = df_kombinirani['Broj videa'].groupby(df_kombinirani['Datum'].dt.to_period('W')).sum()
    for i, v in df_tjedni.items():
        print('index: ', str(i).split('/')[0], 'value: ', v)
        bottom = str(i).split('/')[0]
        bot = bottom.split('-')[2] + '.' + bottom.split('-')[1] + '.'
        top = str(i).split('/')[1]
        to = top.split('-')[2] + '.' + top.split('-')[1] + '.'
        x_datumi.append(bot+' - '+to)
        y_frekvencije.append(v)
    plt.style.use('seaborn')
    plt.plot_date(x_datumi,y_frekvencije,linestyle='solid')
    plt.gcf().autofmt_xdate()
    # date_format = mpl_dates.DateFormatter('%d-%m-%Y')
    # plt.gca().xaxis.set_major_formatter(date_format)
    plt.title('Frekvencije videa')
    plt.xlabel('Datum')
    plt.ylabel('Broj videa')
    plt.tight_layout()
    plt.show()
    return df_tjedni
        
    # - GRAF FREKVENCIJA VIDEA - DANI
    # x_datumi = df_kombinirani['Datum']
    # y_frekvencije = df_kombinirani['Broj videa']
    # plt.style.use('seaborn')
    # plt.plot_date(x_datumi,y_frekvencije,linestyle='solid')
    # plt.gcf().autofmt_xdate()
    # date_format = mpl_dates.DateFormatter('%d-%m-%Y')
    # plt.gca().xaxis.set_major_formatter(date_format)
    # plt.title('Frekvencije videa')
    # plt.xlabel('Datum')
    # plt.ylabel('Broj videa')
    # plt.tight_layout()
    # plt.show()
    
    df_naslovi = pd.DataFrame(df_kombinirani)
    df_naslovi.to_csv('ObjaveVidea.csv',index = False, encoding='utf-8-sig')
    
    # return df_naslovi
    
    komentar_datum = list(zip(komentar,objava_komentara))
    komentari_sort = sorted(komentar_datum, key=sort_date, reverse=False)
   
    datumi_kom = [datum[1] for datum in komentari_sort]
    frekvencije_datuma_kom = []
    for datum in datumi_kom:
        frekvencije_datuma_kom.append(datumi_kom.count(datum))
    frekvencije_kom = list(zip(datumi_kom,frekvencije_datuma_kom))
    jedinstvene_frekvencije_kom = list(set(frekvencije_kom))
    
    c = collections.defaultdict(list)
    for a,b in komentari_sort:
        c[b].extend([a])  # add to existing list or create a new one
    
    datum_komentar = list(c.items())
    
    df3 = pd.DataFrame(jedinstvene_frekvencije_kom,columns=['Datum','Broj komentara'])
    df4 = pd.DataFrame(datum_komentar,columns=['Datum','Komentar'])
    df_kom = pd.merge(df3, df4, on="Datum")

    df_kom['Datum'] = pd.to_datetime(df_kom['Datum'])
    df_kom.sort_values('Datum',inplace=True)
    
    df_komentari = pd.DataFrame(df_kom)
    df_komentari.to_csv('ObjaveKomentara.csv',index = False, encoding='utf-8-sig')
    
    x_datumi_kom = df_kom['Datum']
    y_frekvencije_kom = df_kom['Broj komentara']
    
    plt.style.use('seaborn')
    plt.plot_date(x_datumi_kom,y_frekvencije_kom,linestyle='solid')
    plt.gcf().autofmt_xdate()
    date_format = mpl_dates.DateFormatter('%d-%m-%Y')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.title('Frekvencije komentara')
    plt.xlabel('Datum')
    plt.ylabel('Broj komentara')
    plt.tight_layout()
    plt.show()
   
    print(sorted(jedinstvene_frekvencije_kom, key=sort_date_0, reverse=False))
    return jedinstvene_frekvencije_kom
    
if __name__ == '__main__':
    # df_comb = pearsonova_korelacija()
    # df = korelacija()
    # frekvencije,korpus = frekvencije_rijeci()
    df_videa = objave_po_danima()
    #df1 = concat_dataframes()
    # df = micanje_duplikata()
    # df2 = detekcija_jezika()
    # views,likes,comments = popularna_videa()
    # graph_df = lista_bridova()
    # stupanj2,međupoloženost2,blizina2,vektor2 = analiza_mreze()
    
