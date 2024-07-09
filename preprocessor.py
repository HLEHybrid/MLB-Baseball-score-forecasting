import pybaseball
import pandas as pd
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# 각 선수에 대해 key_mlbam 값을 도출하는 함수
def get_mlbam_key(df):

    player_id_list = list(df['IDfg'])
    player_data = pybaseball.playerid_reverse_lookup(player_id_list, key_type='fangraphs')
    player_data = player_data[['key_fangraphs','key_mlbam']]
    new_column_names = {'key_fangraphs': 'IDfg'}
    player_data.rename(columns=new_column_names, inplace=True)

    df = pd.merge(df,player_data)

    return df

def get_bbref_key(df):

    player_id_list = list(df['IDfg'])
    player_data = pybaseball.playerid_reverse_lookup(player_id_list, key_type='fangraphs')
    player_data = player_data[['key_fangraphs','key_bbref']]
    new_column_names = {'key_fangraphs': 'IDfg'}
    player_data.rename(columns=new_column_names, inplace=True)

    df = pd.merge(df,player_data)

    return df


def bat_recode(year):
    bat_df = pybaseball.batting_stats(year, qual=30)

    bat_df = get_bbref_key(bat_df)
    bat_df = get_mlbam_key(bat_df)

    # 선수들의 스플릿 데이터를 멀티프로세싱을 통해 수집
    splits_df = collect_player_split(list(bat_df['IDfg']))
    splits_df['Bats'] = splits_df['Bats'].apply(split_to_num)
    splits_df['Throws'] = splits_df['Throws'].apply(split_to_num)
    
    bat_df['bat_split'] = list(splits_df['Bats'])

    batting_stats_columns = ['IDfg', 'key_mlbam', 'key_bbref', 'Name', 'Team', 'bat_split', 'GB/FB', 'LD%', 'GB%', 'FB%', 'IFFB%', 'HR/FB', 'Spd', 'BsR', 'wFB/C', 'wSL/C', 'wCT/C', 'wCB/C', 'wCH/C', 'wSF/C', 'wKN/C', 'O-Swing%', 'Z-Swing%', 'Swing%', 
    'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%', 'SwStr%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%', 'wFA/C (sc)', 'wFC/C (sc)', 'wFS/C (sc)', 
    'wFO/C (sc)', 'wSI/C (sc)', 'wSL/C (sc)', 'wCU/C (sc)', 'wKC/C (sc)', 'wCH/C (sc)', 'wKN/C (sc)', 'O-Swing% (sc)', 'Z-Swing% (sc)', 'Swing% (sc)', 'O-Contact% (sc)',
    'Z-Contact% (sc)', 'Contact% (sc)', 'Zone% (sc)', 'LD+%', 'GB%+' ,'FB%+', 'HR/FB%+', 'Pull%+', 'Cent%+', 'Oppo%+', 'Soft%+', 'Med%+', 'Hard%+', 'EV', 'LA', 'Barrel%', 'maxEV', 
    'HardHit%', 'CStr%', 'CSW%', 'wCH/C (pi)', 'wCU/C (pi)', 'wFA/C (pi)', 'wFC/C (pi)', 'wFS/C (pi)', 'wKN/C (pi)', 'wSI/C (pi)', 'wSL/C (pi)', 'O-Swing% (pi)', 
    'Z-Swing% (pi)', 'Swing% (pi)', 'O-Contact% (pi)', 'Z-Contact% (pi)', 'Contact% (pi)', 'Zone% (pi)', 'Pace', 'UBR']

    bat_df = bat_df[batting_stats_columns]

    # 열 이름 변경
    new_column_names = {'player_id': 'key_mlbam'}

    statcast_batter_exitvelo_barrels_data = pybaseball.statcast_batter_exitvelo_barrels(year, 10)
    statcast_batter_exitvelo_barrels_data_columns = ['player_id', 'anglesweetspotpercent', 'ev50', 'fbld', 'max_distance', 'avg_distance', 'avg_hr_distance', 'ev95percent']
    statcast_batter_exitvelo_barrels_data = statcast_batter_exitvelo_barrels_data[statcast_batter_exitvelo_barrels_data_columns]

    statcast_batter_exitvelo_barrels_data.rename(columns=new_column_names, inplace=True)
    bat_df_merge = pd.merge(bat_df, statcast_batter_exitvelo_barrels_data, on='key_mlbam', how='left')

    new_column_names = {'key_mlbam': 'batter_key_mlbam'}
    bat_df_merge.rename(columns=new_column_names, inplace=True)

    # 열의 데이터를 숫자로 변환
    # for col in bat_df_merge.columns[5:]:
    #     bat_df_merge[col] = pd.to_numeric(bat_df_merge[col], errors='coerce')
    
    # 결측치 처리
    for col in range(bat_df_merge.shape[1]-5):
        column_data = bat_df_merge.iloc[:, col+5]
        valid_data = column_data[~pd.isna(column_data)]
        Q1 = np.nanquantile(valid_data, 0.25)
        Q3 = np.nanquantile(valid_data, 0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 이상치를 제외한 최솟값과 최댓값 계산
        filtered_data = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)]
        filtered_min = np.min(filtered_data)

        # 결측치 대체 전략 설정
        replacement_value = filtered_min if filtered_min < Q3 else Q3

        # 결측치 대체
        bat_df_merge.iloc[pd.isna(bat_df_merge.iloc[:, col+5]), col+5] = replacement_value

    # bat_df_merge.to_excel('batting.xlsx')

    # 리스트 컴프리헨션을 사용하는 방법
    bat_df_merge_columns = [col for col in bat_df_merge.columns if col not in ['IDfg', 'batter_key_mlbam', 'key_bbref', 'Name', 'Team']]

    return bat_df_merge, bat_df_merge_columns

def pitch_recode(year):
    pitching_df = pybaseball.pitching_stats(year, qual=10)

    pitching_df = get_bbref_key(pitching_df)
    pitching_df = get_mlbam_key(pitching_df)

    # 선수들의 스플릿 데이터를 멀티프로세싱을 통해 수집
    splits_df = collect_player_split(list(pitching_df['IDfg']))
    splits_df['Bats'] = splits_df['Bats'].apply(split_to_num)
    splits_df['Throws'] = splits_df['Throws'].apply(split_to_num)
    
    pitching_df['pitch_split'] = list(splits_df['Throws'])

    pitching_stats_columns = ['IDfg', 'key_mlbam', 'key_bbref', 'Name', 'Team', 'pitch_split', 'GB/FB', 'LD%', 'GB%', 'FB%', 'IFFB%', 'HR/FB', 'FB% 2', 'FBv', 'SL%', 'SLv', 'CT%', 'CTv', 'CB%', 'CBv', 'CH%', 'CHv', 'SF%', 'SFv', 'KN%', 'KNv', 'wFB/C', 'wSL/C', 
    'wCT/C', 'wCB/C', 'wCH/C', 'wSF/C', 'wKN/C', 'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%', 'SwStr%', 'FA% (sc)', 
    'FC% (sc)', 'FS% (sc)', 'FO% (sc)', 'SI% (sc)', 'SL% (sc)', 'CU% (sc)', 'KC% (sc)', 'CH% (sc)', 'KN% (sc)', 'vFA (sc)', 'vFC (sc)', 'vFS (sc)', 'vFO (sc)', 
    'vSI (sc)', 'vSL (sc)', 'vCU (sc)', 'vKC (sc)', 'vCH (sc)', 'vKN (sc)', 'FA-X (sc)', 'FC-X (sc)', 'FS-X (sc)', 'FO-X (sc)', 'SI-X (sc)', 'SL-X (sc)', 'CU-X (sc)', 'KC-X (sc)', 
    'CH-X (sc)', 'KN-X (sc)', 'FA-Z (sc)', 'FC-Z (sc)', 'FS-Z (sc)', 'FO-Z (sc)', 'SI-Z (sc)', 'SL-Z (sc)', 'CU-Z (sc)', 'KC-Z (sc)', 'CH-Z (sc)',
    'KN-Z (sc)', 'wFA/C (sc)', 'wFC/C (sc)', 'wFS/C (sc)', 'wFO/C (sc)', 'wSI/C (sc)', 'wSL/C (sc)', 'wCU/C (sc)', 'wKC/C (sc)', 'wCH/C (sc)', 'wKN/C (sc)', 
    'O-Swing% (sc)', 'Z-Swing% (sc)', 'Swing% (sc)', 'O-Contact% (sc)', 'Z-Contact% (sc)', 'Contact% (sc)', 'Zone% (sc)', 'LD%+', 'GB%+', 'FB%+', 'HR/FB%+', 'Pull%+', 'Cent%+', 
    'Oppo%+', 'Soft%+', 'Med%+', 'Hard%+', 'EV', 'LA', 'Barrel%', 'maxEV', 'HardHit%', 'CStr%', 'CSW%', 'botOvr CH', 'botStf CH', 'botCmd CH', 'botOvr CU', 'botStf CU', 'botCmd CU', 
    'botOvr FA', 'botStf FA', 'botCmd FA', 'botOvr SI', 'botStf SI', 'botCmd SI', 'botOvr SL', 'botStf SL', 'botCmd SL', 'botOvr KC', 'botStf KC', 'botCmd KC', 'botOvr FC', 'botStf FC', 
    'botCmd FC', 'botOvr FS', 'botStf FS', 'botCmd FS', 'botOvr', 'botStf', 'botCmd', 'botxRV100', 'Stf+ CH', 'Loc+ CH', 'Pit+ CH', 'Stf+ CU', 'Loc+ CU', 'Pit+ CU', 'Stf+ FA', 'Loc+ FA',
    'Pit+ FA', 'Stf+ SI', 'Loc+ SI', 'Pit+ SI', 'Stf+ SL', 'Loc+ SL', 'Pit+ SL', 'Stf+ KC', 'Loc+ KC', 'Pit+ KC', 'Stf+ FC', 'Loc+ FC', 'Pit+ FC', 'Stf+ FS', 'Loc+ FS', 'Pit+ FS', 'Stf+ FO', 'Loc+ FO', 
    'Pit+ FO', 'Stuff+', 'Location+', 'Pitching+', 'CH% (pi)', 'CU% (pi)', 'FA% (pi)', 'FC% (pi)', 'FS% (pi)', 'KN% (pi)', 'SI% (pi)', 'SL% (pi)', 'vCH (pi)', 
    'vCU (pi)', 'vFA (pi)', 'vFC (pi)', 'vFS (pi)', 'vKN (pi)', 'vSI (pi)', 'vSL (pi)', 'CH-X (pi)', 'CU-X (pi)', 'FA-X (pi)', 'FC-X (pi)', 'FS-X (pi)', 'KN-X (pi)', 
    'SI-X (pi)', 'SL-X (pi)', 'CH-Z (pi)', 'CU-Z (pi)', 'FA-Z (pi)', 'FC-Z (pi)', 'FS-Z (pi)', 'KN-Z (pi)', 'SI-Z (pi)', 'SL-Z (pi)', 'wCH/C (pi)', 'wCU/C (pi)', 
    'wFA/C (pi)', 'wFC/C (pi)', 'wFS/C (pi)', 'wKN/C (pi)', 'wSI/C (pi)', 'wSL/C (pi)', 'O-Swing% (pi)', 'Z-Swing% (pi)', 'Swing% (pi)', 'O-Contact% (pi)', 'Z-Contact% (pi)', 
    'Contact% (pi)', 'Zone% (pi)','Pace']

    pitching_df = pitching_df[pitching_stats_columns]

    # 열 이름 변경
    statcast_new_column_names = {'player_id': 'key_mlbam'}

    statcast_pitcher_exitvelo_barrels_data = pybaseball.statcast_pitcher_exitvelo_barrels(year, 10)
    statcast_pitcher_exitvelo_barrels_columns = ['player_id', 'anglesweetspotpercent', 'ev50', 'fbld', 'max_distance', 'avg_distance', 'avg_hr_distance', 'ev95percent']
    statcast_pitcher_exitvelo_barrels_data = statcast_pitcher_exitvelo_barrels_data[statcast_pitcher_exitvelo_barrels_columns]
    statcast_pitcher_exitvelo_barrels_data.rename(columns=statcast_new_column_names, inplace=True)

    pitcher_df_merge = pd.merge(pitching_df, statcast_pitcher_exitvelo_barrels_data, on='key_mlbam', how='left')

    pitcher_df_merge.columns = ['(P) ' + column for column in pitcher_df_merge.columns]

    new_column_names = {'(P) key_mlbam': 'pitcher_key_mlbam'}
    pitcher_df_merge.rename(columns=new_column_names, inplace=True)

    # 열의 데이터를 숫자로 변환
    # for col in pitcher_df_merge.columns[5:]:
    #     pitcher_df_merge[col] = pd.to_numeric(pitcher_df_merge[col], errors='coerce')
    
    # 결측치 처리
    for col in range(pitcher_df_merge.shape[1]-5):
        column_data = pitcher_df_merge.iloc[:, col+5]
        valid_data = column_data[~pd.isna(column_data)]
        Q1 = np.nanquantile(valid_data, 0.25)
        Q3 = np.nanquantile(valid_data, 0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 이상치를 제외한 최솟값과 최댓값 계산
        filtered_data = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)]
        filtered_min = np.min(filtered_data)

        # 결측치 대체 전략 설정
        replacement_value = filtered_min if filtered_min < Q3 else Q3

        # 결측치 대체
        pitcher_df_merge.iloc[pd.isna(pitcher_df_merge.iloc[:, col+5]), col+5] = replacement_value
    
    # pitcher_df_merge.to_excel('pitching.xlsx')
    
    pitcher_df_merge_columns = [col for col in pitcher_df_merge.columns if col not in ['(P) IDfg', 'pitcher_key_mlbam', '(P) key_bbref', '(P) Name', '(P) Team']]

    return pitcher_df_merge, pitcher_df_merge_columns

def fielding_recode(year, home_away):
    fielding_data = pybaseball.team_fielding(year)
    # 나눌 지표들 리스트
    metrics = ['rSZ', 'rCERA', 'rTS', 'rSB', 'rGDP', 'rARM', 'rGFP', 'rPM', 'DRS', 
           'ARM', 'DPR', 'RngR', 'ErrR', 'UZR', 'Def', 'FRM', 'OAA', 'Range']

    # 각 지표를 경기 수(G)로 나눈 값을 새로운 열로 추가
    for metric in metrics:
        new_column_name = f"{metric}/G"
        fielding_data[new_column_name] = fielding_data[metric] / fielding_data['G']

    team_short_names = {'Rockies':'COL', 'Red Sox':'BOS', 'Royals':'KC', 'Reds':'CIN', 'Rangers':'TEX', 'Nationals':'WSH', 
                        'Angels':'LAA', 'Cardinals':'STL', 'Astros':'HOU', 'Braves':'ATL', 'Phillies':'PHI', 'Twins':'MIN', 
                        'Blue Jays':'TOR', 'Diamondbacks':'AZ', 'Cubs':'CHC', 'Pirates':'PIT', 'Marlins':'MIA', 'White Sox':'CWS',
                         'Dodgers':'LAD', 'Brewers':'MIL', 'Yankees':'NYY', 'Orioles':'BAL', 'Tigers':'DET', 'Athletics':'OAK', 
                         'Rays':'TB', 'Guardians':'CLE', 'Giants':'SF', 'Padres':'SD', 'Mets':'NYM', 'Mariners':'SEA','Indians':'CLE'}
    
    fielding_data['Short name'] = fielding_data['Team'].map(team_short_names)
    
    if home_away == 'home':
        fielding_new_column_names = {'Short name': 'home_team'}
        fielding_data.rename(columns=fielding_new_column_names, inplace=True)
        fielding_data_columns = [f'{metric}/G' for metric in metrics]
        fielding_data = fielding_data[['home_team'] + fielding_data_columns]

    elif home_away == 'away':
        fielding_new_column_names = {'Short name': 'away_team'}
        fielding_data.rename(columns=fielding_new_column_names, inplace=True)
        fielding_data_columns = [f'{metric}/G' for metric in metrics]
        fielding_data = fielding_data[['away_team'] + fielding_data_columns]
    
    # fielding_data.to_excel('fielding.xlsx')
    
    return fielding_data, fielding_data_columns

def gamelog_agg(year):
    batting_player_data, batting_player_data_columns = bat_recode(year)
    pitching_player_data, pitching_player_data_columns = pitch_recode(year)
    home_fielder_data, fielder_data_columns = fielding_recode(year, 'home')
    away_fielder_data, _ = fielding_recode(year, 'away')

    start_date = str(year) + '-01-01'
    end_date = str(year) + '-12-31'

    gamelog = pybaseball.statcast(start_dt=start_date, end_dt=end_date)

    gamelog_columns = ['game_date', 'home_team', 'away_team', 'batter', 'pitcher', 'events', 'game_type', 'inning_topbot']

    gamelog = gamelog[gamelog_columns]
    gamelog = gamelog[gamelog['game_type'] == 'R']
    gamelog = gamelog.dropna(subset=['events'])

    park_factors = {'COL':112, 'BOS':107, 'KC':105, 'CIN':104, 'TEX':102, 'WSH':102, 'LAA':101, 'STL':101, 'HOU':101,
                    'ATL':101, 'PHI':101, 'MIN':101, 'TOR':100, 'AZ':100, 'CHC':100, 'PIT':100, 'MIA':100, 'CWS':99, 
                    'LAD':99, 'MIL':99, 'NYY':99, 'BAL':98, 'DET':98, 'OAK':97, 'TB':97, 'CLE':96, 'SF':96, 'SD':96, 
                    'NYM':95, 'SEA':92}

    gamelog['park_factor'] = gamelog['home_team'].map(park_factors)

    new_column_names = {'batter': 'batter_key_mlbam', 'pitcher': 'pitcher_key_mlbam'}
    gamelog.rename(columns=new_column_names, inplace=True)


    gamelog_agg = pd.merge(gamelog, batting_player_data)
    gamelog_agg = pd.merge(gamelog_agg,pitching_player_data)

    # 먼저 gamelog_agg를 home_fielder_data와 away_fielder_data에 공통된 키로 merge할 수 있는 데이터를 추출합니다.
    gamelog_agg_home = gamelog_agg[gamelog_agg['inning_topbot'] == 'Top']
    gamelog_agg_away = gamelog_agg[gamelog_agg['inning_topbot'] == 'Bot']

    # 각각을 별도로 merge합니다.
    gamelog_agg_home = pd.merge(gamelog_agg_home, home_fielder_data, on='home_team')
    gamelog_agg_away = pd.merge(gamelog_agg_away, away_fielder_data, on='away_team')

    # 다시 합칩니다.
    gamelog_agg = pd.concat([gamelog_agg_home, gamelog_agg_away], ignore_index=True)

    # gamelog_agg.to_excel('gamelog.xlsx')


    result_list = []

    for result in list(gamelog_agg['events']):
        if 'field_out' in result:
            result_list.append('Out')
        elif 'double_play' in result:
            result_list.append('DoublePlay')
        elif 'error' in result:
            result_list.append('Single')
        elif 'single' in result:
            result_list.append('Single')
        elif 'double' in result:
            result_list.append('Double')
        elif 'triple' in result:
            result_list.append('Triple')
        elif 'home_run' in result:
            result_list.append('HomeRun')
        elif 'out' in result:
            result_list.append('Out')
        elif 'strikeout' in result:
            result_list.append('Out')
        elif 'sac_fly' in result:
            result_list.append('Out')
        elif 'fielders_choice' in result:
            result_list.append('Out')
        elif 'hit_by_pitch' in result:
            result_list.append('Walk')
        elif 'walk' in result:
            result_list.append('Walk')
        else:
            result_list.append(None)

    gamelog_agg['Result'] = result_list

    gamelog_agg = gamelog_agg[['game_date', 'home_team', 'away_team', 'batter_key_mlbam', 'pitcher_key_mlbam', 'IDfg', '(P) IDfg', 'key_bbref', '(P) key_bbref', 
                               'Name', '(P) Name', 'Team', '(P) Team'] + batting_player_data_columns + pitching_player_data_columns 
                               + fielder_data_columns + ['park_factor','Result']]

    gamelog_agg = gamelog_agg.dropna(subset=['Result'])

    with pd.ExcelWriter('gamelog_agg_' + str(year) + '.xlsx', engine='xlsxwriter') as writer:
        writer.book.use_zip64()
        gamelog_agg.to_excel(writer, index=False)

    return gamelog_agg


# def get_player_split_from_fangraphs(player_ids):
#     player_data = pybaseball.playerid_reverse_lookup(player_ids, key_type='fangraphs')
#     all_info = []
#     for i in range(len(player_ids)):
    
#         try:
#             player_id = player_data['key_fangraphs'].values[i]
#             first_name = player_data['name_first'].values[i]
#             last_name = player_data['name_last'].values[i]
#             url = f"https://www.fangraphs.com/players/{first_name}-{last_name}/{player_id}/stats"
#             response = requests.get(url)
#             if response.status_code != 200:
#                 print(f"Failed to fetch data for player {player_id}")
#                 player_info = {}
#                 player_info['IDfg'] = player_id
#                 player_info['Bats'] = 0.0
#                 player_info['Throws'] = 0.0
#                 all_info.append(player_info)
#                 continue

#             print(player_id)
#             dom = BeautifulSoup(response.content, 'html.parser')
#             player_info = {}
#             player_info['IDfg'] = player_id
#             player_info['Bats'] = dom.find_all(attrs={'class':'header_item__6AFbi'})[2].text[-3]
#             player_info['Throws'] = dom.find_all(attrs={'class':'header_item__6AFbi'})[2].text[-1]

#             all_info.append(player_info)
        
#         except Exception as e:
#             print(f"Error fetching splits for player {player_id}: {e}")
#             player_info = {}
#             player_info['IDfg'] = player_id
#             player_info['Bats'] = 0.0
#             player_info['Throws'] = 0.0
#             all_info.append(player_info)

#     return pd.DataFrame(all_info)

def get_player_split_from_fangraphs(player_id):
    try:
        player_data = pybaseball.playerid_reverse_lookup([player_id], key_type='fangraphs')
        first_name = player_data['name_first'].values[0]
        last_name = player_data['name_last'].values[0]
        url = f"https://www.fangraphs.com/players/{first_name}-{last_name}/{player_id}/stats"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data for player {player_id}")
            player_info = {'IDfg': player_id, 'Bats': 'R', 'Throws': 'R'}
            return player_info

        dom = BeautifulSoup(response.content, 'html.parser')
        player_info = {'IDfg': player_id}
        player_info['Bats'] = dom.find_all(attrs={'class':'header_item__6AFbi'})[2].text[-3]
        player_info['Throws'] = dom.find_all(attrs={'class':'header_item__6AFbi'})[2].text[-1]

        return player_info
    
    except Exception as e:
        print(f"Error fetching splits for player {player_id}: {e}")
        return {'IDfg': player_id, 'Bats': 'R', 'Throws': 'R'}

def collect_player_split(player_ids):
    all_info = []

    with ThreadPoolExecutor() as thread_executor:
        future_to_player = {thread_executor.submit(get_player_split_from_fangraphs, player_id): player_id for player_id in player_ids}
        
        for future in as_completed(future_to_player):
            player_id = future_to_player[future]
            try:
                result = future.result()
                if result is not None:
                    all_info.append(result)
            except Exception as e:
                print(f"Error processing player {player_id}: {e}")

    return pd.DataFrame(all_info)

def split_to_num(split):
    if split == 'R':
        return 0.0
    elif split == 'L':
        return 1.0
    elif split == 'B':
        return 0.5
    elif split == 'S':
        return 0.5
    else:
        return 0.0


if __name__ == '__main__':
    # 분석을 원하는 연도 입력
    years = [2022,2023]
    gamelog_all = pd.DataFrame()

    for year in years:
        gamelog = gamelog_agg(year)
        gamelog_all = pd.concat([gamelog_all,gamelog],axis=0)

    gamelog_all.dropna(axis=1, how='all', inplace=True)
    
    for col in range(gamelog_all.shape[1]-15):
        column_data = gamelog_all.iloc[:, col+14]
        valid_data = column_data[~pd.isna(column_data)]
        Q1 = np.nanquantile(valid_data, 0.25)
        Q3 = np.nanquantile(valid_data, 0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 이상치를 제외한 최솟값과 최댓값 계산
        filtered_data = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)]
        filtered_min = np.min(filtered_data)

        # 결측치 대체 전략 설정
        replacement_value = filtered_min if filtered_min < Q3 else Q3

        # 결측치 대체
        gamelog_all.iloc[pd.isna(gamelog_all.iloc[:, col+14]), col+14] = replacement_value

    gamelog_all.to_pickle('gamelog_agg.pkl')
    # Use xlsxwriter with ZIP64 extension enabled
    # with pd.ExcelWriter('gamelog_agg.xlsx', engine='xlsxwriter') as writer:
    #    writer.book.use_zip64()
    #    gamelog_all.to_excel(writer, index=False)
