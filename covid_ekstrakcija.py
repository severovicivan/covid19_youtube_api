import os
import re
import pickle
import pandas as pd
import google.oauth2.credentials

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret.
CLIENT_SECRETS_FILE = "client_secret_1.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

def get_authenticated_service():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    #  Check if the credentials are invalid or do not exist
    if not credentials or not credentials.valid:
        # Check if the credentials have expired
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()
 
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
 
    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

def get_videos(service, **kwargs):
    final_results = []
    results = service.search().list(**kwargs).execute()
 
    i = 0
    max_pages = 100
    while results and i < max_pages:
        final_results.extend(results['items'])
 
        # Check if another page exists
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.search().list(**kwargs).execute()
            i += 1
        else:
            print('Broj nađenih stranica: ' + str(i))
            break
 
    return final_results

def search_videos_by_keyword(service, **kwargs):
    results = get_videos(service, **kwargs)
    print('Broj pronađenih videa je: ' + str(len(results)))
    
    stored_videos = 0
    
    df_all_videos = []
    
    video_id = []
    channel = []
    video_title = []
    video_desc = []
    # print(results)

    for item in results:
        video_id = item['id']['videoId']
        channel = item['snippet']['channelTitle']   
        video_title = item['snippet']['title']
        
        print('\nNaslov videa: ' + video_title)
        
        video_desc = item['snippet']['description']
        publish_date = item['snippet']['publishedAt']
        
        try:
            stats = service.videos().list(
                    part='statistics',
                    id=item['id']['videoId']).execute()
        except:
            print('Nema statistike za ' + video_title)
        
        try:
            views_count = stats['items'][0]['statistics']['viewCount']
        except:
            views_count = 0
            print('Nema pregleda za ' + video_title)
        try:
            likes_count = stats['items'][0]['statistics']['likeCount']
        except:
            likes_count = 0
            print('Nema lajkova za ' + video_title)
        try:
            dislikes_count = stats['items'][0]['statistics']['dislikeCount']
        except:
            dislikes_count = 0
            print('Nema dislajkova za ' + video_title)
        try:
            comments_count = stats['items'][0]['statistics']['commentCount']
        except:
            comments_count = 0
            print('Nema komentara za ' + video_title)
        
        channel = channel.replace('š', 's').replace('Š', 'S').replace('đ', 'd').replace('Đ', 'D').replace('dž','d').replace('DŽ','D').replace('č', 'c').replace('Č', 'C').replace('ć', 'c').replace('Ć', 'C').replace('ž', 'z').replace('Ž', 'Z')
        video_title = video_title.replace('š', 's').replace('Š', 'S').replace('đ', 'd').replace('Đ', 'D').replace('dž','d').replace('DŽ','D').replace('č', 'c').replace('Č', 'C').replace('ć', 'c').replace('Ć', 'C').replace('ž', 'z').replace('Ž', 'Z')
        video_desc = video_desc.replace('š', 's').replace('Š', 'S').replace('đ', 'd').replace('Đ', 'D').replace('dž','d').replace('DŽ','D').replace('č', 'c').replace('Č', 'C').replace('ć', 'c').replace('Ć', 'C').replace('ž', 'z').replace('Ž', 'Z')
        
        video_id_pop = []
        channel_pop = []
        video_title_pop = []
        video_desc_pop = []
        comments_pop = []
        comment_id_pop = []
        authors_pop = []
        authors_cpop = []
        published_pop = []
        reply_count_pop = []
        like_count_pop = []

        comments_temp = []
        comment_id_temp = []
        authors_temp = []
        authors_channel = []
        published_temp = []
        reply_count_temp = []
        like_count_temp = []
        
        if int(comments_count) > 0:
            response = service.commentThreads().list(
                part='snippet', videoId=item['id']['videoId'],
                maxResults = 10, order='relevance', textFormat='plainText'
            ).execute()

            for comment in response['items']:
                comments_temp.append(comment['snippet']['topLevelComment']['snippet']['textDisplay'].
                                 replace('š', 's').replace('Š', 'S').
                                 replace('đ', 'd').replace('Đ', 'D').
                                 replace('dž','d').replace('DŽ','D').
                                 replace('č', 'c').replace('Č', 'C').
                                 replace('ć', 'c').replace('Ć', 'C').
                                 replace('ž', 'z').replace('Ž', 'Z'))
                authors_temp.append(comment['snippet']['topLevelComment']['snippet']['authorDisplayName'].
                                 replace('š', 's').replace('Š', 'S').
                                 replace('đ', 'd').replace('Đ', 'D').
                                 replace('dž','d').replace('DŽ','D').
                                 replace('č', 'c').replace('Č', 'C').
                                 replace('ć', 'c').replace('Ć', 'C').
                                 replace('ž', 'z').replace('Ž', 'Z'))
                authors_channel.append(comment['snippet']['topLevelComment']['snippet']['authorChannelId'])
                published_temp.append(comment['snippet']['topLevelComment']['snippet']['publishedAt'])
                comment_id_temp.append(comment['snippet']['topLevelComment']['id'])
                reply_count_temp.append(comment['snippet']['totalReplyCount'])
                like_count_temp.append(comment['snippet']['topLevelComment']['snippet']['likeCount'])
                comments_pop.extend(comments_temp)
                authors_pop.extend(authors_temp)
                authors_cpop.extend(authors_channel)
                
                published_pop.extend(published_temp)
                comment_id_pop.extend(comment_id_temp)
                reply_count_pop.extend(reply_count_temp)
                like_count_pop.extend(like_count_temp)
                    
                video_id_pop.extend([video_id]*len(comments_temp))
                channel_pop.extend([channel]*len(comments_temp))
                video_title_pop.extend([video_title]*len(comments_temp))
                video_desc_pop.extend([video_desc]*len(comments_temp))

            print(comments_temp)

            output_dict = {
                'Channel': channel_pop,
                'Video Title': video_title_pop,
                'Video Published': publish_date,
                'Video Views': views_count,
                'Video Comments': comments_count,
                'Video Likes': likes_count,
                'Video Dislikes': dislikes_count,
                'Video Description': video_desc_pop,
                'Video ID': video_id_pop,
                'Comment': comments_pop,
                'Author': authors_pop,
                'Comment Published' : published_pop,
                'Comment ID': comment_id_pop,
                'Replies': reply_count_pop,
                'Comment Likes': like_count_pop,
            }

            output_df = pd.DataFrame(output_dict, columns = output_dict.keys())
            df_for_join = pd.DataFrame(output_dict)
            unique_df = output_df.drop_duplicates(subset=['Comment'])
            unique_join = df_for_join.drop_duplicates(subset=['Comment'])
            
            df_all_videos.append(unique_join)
            
            print(unique_df.head())
            try:
                unique_df.to_csv(video_title +".csv",index = False)
            except:
                print('CSV '+ video_title +' već postoji')
                
            stored_videos += 1
            
        else:
            print('Video nema komentara')
            
    print('Broj videa s bar jednim komentarom: ' + str(stored_videos))
    
    
    try:
        df_merged = pd.concat(df_all_videos)
        return df_merged
    except:
        print('Nema videa za spajanje')
    
    
    # for item in results['items']:
    #     print('%s - %s' % (item['snippet']['title'], item['id']['videoId']))
    #     number_of_videos += 1
    # print('Broj pronađenih videa je: ' + str(number_of_videos))

if __name__ == '__main__':
    # When running locally, disable OAuthlib's HTTPs verification. When
    # running in production *do not* leave this option enabled.
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    service = get_authenticated_service()
    videos = []
    # virus,cjepivo,zavjera,imunitet,simptomi,mjere,stožer....maske treba dodat
    keywords = ['korona virus']
    for word in keywords:
    # try:
        dataframe = search_videos_by_keyword(service, q=word,
        part='id,snippet', order='relevance', type='video',
        relevanceLanguage='hr', safeSearch='none',
        publishedAfter='2020-01-01T00:00:00Z',
        location='45.749189055470914, 16.61218970840497',
        locationRadius='50km',
        regionCode='HR')
        
        try:
            dataframe['Keyword'] = word
        except:
            print('Ne želi dodati stupac ' + word)
        videos.append(dataframe)
    # except:
    #     print('Nema podataka za riječ ' + word)
        
    result = pd.concat(videos)
    df_videos = pd.DataFrame(result,columns=['Channel','Video Title','Video Published','Video Views','Video Comments','Video Likes','Video Dislikes','Video Description','Video ID','Comment','Author','Comment Published','Comment ID','Replies','Comment Likes','Keyword'])    
    # if file does not exist write header 
    if not os.path.isfile('ListaVidea.csv'):
       df_videos.to_csv('ListaVidea.csv',index=False, encoding='utf-8-sig')
    else: # else it exists so append without writing the header
       # df.to_csv('ListaVidea.csv', mode='a', header=False)
       df_videos.to_csv('ListaVidea.csv',index=False, mode='a', encoding='utf-8-sig', header=False)
