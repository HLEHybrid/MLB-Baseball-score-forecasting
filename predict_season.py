import pandas as pd
import time
import numpy as np
import preprocessor
from pickle import load
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import AnchoredText
import math
import pybaseball
from bs4 import BeautifulSoup
import requests
import koreanize_matplotlib
from datetime import datetime, timedelta

# 마르코프 체인에서 상태 ID를 반환하는 함수
def getID(first, second, third, outs, inning):
    """
    주어진 파라미터로 상태 ID를 반환합니다.
    :param first: 1루에 주자가 있는지 여부 (0 또는 1)
    :param second: 2루에 주자가 있는지 여부 (0 또는 1)
    :param third: 3루에 주자가 있는지 여부 (0 또는 1)
    :param outs: 아웃 수 (0, 1, 2)
    :param inning: 이닝 수 (1-9)
    :returns: int. 주어진 파라미터로 계산된 상태 ID
    """
    return first + 2 * second + 4 * third + 8 * outs + 24 * (inning - 1)


# 마르코프 체인의 상태를 나타내는 클래스
class State:
    """
    마르코프 체인에서 상태를 나타내는 클래스입니다.
    """

    def __init__(self, stateID):
        self.id = stateID
        if stateID == 216:
            self.i = 9
            self.o = 3
            self.t = 0
            self.s = 0
            self.f = 0
        else:
            self.i = (stateID // 24) + 1
            stateID -= (self.i - 1) * 24
            self.o = stateID // 8
            stateID -= self.o * 8
            self.t = stateID // 4
            stateID -= self.t * 4
            self.s = stateID // 2
            stateID -= self.s * 2
            self.f = stateID

    # 주자가 진루하는 상황들에 대한 함수들
    def walk(self):
        """
        타자가 걸어 나가는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        if self.f == 1:
            if self.s == 1:
                if self.t == 1:
                    return (getID(1, 1, 1, self.o, self.i), 1)
                else:
                    return (getID(1, 1, 1, self.o, self.i), 0)
            else:
                return (getID(1, 1, self.t, self.o, self.i), 0)
        else:
            return (getID(1, self.s, self.t, self.o, self.i), 0)

    def single(self):
        """
        타자가 단타를 치는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        return (getID(1, self.f, self.s, self.o, self.i), self.t)

    def double(self):
        """
        타자가 2루타를 치는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        return (getID(0, 1, self.f, self.o, self.i), self.s + self.t)

    def triple(self):
        """
        타자가 3루타를 치는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        return (getID(0, 0, 1, self.o, self.i), self.f + self.s + self.t)

    def homeRun(self):
        """
        타자가 홈런을 치는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        return (getID(0, 0, 0, self.o, self.i), 1 + self.f + self.s + self.t)

    def out(self):
        """
        타자가 아웃되는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        if self.o == 2:
            return (getID(0, 0, 0, 0, self.i + 1), 0)
        else:
            return (getID(self.f, self.s, self.t, self.o + 1, self.i), 0)

    def doublePlay(self):
        """
        타자가 병살타를 치는 상황
        :returns: (int, int). 새로운 상태 ID와 득점 수
        """
        if self.o >= 1:
            return (getID(0, 0, 0, 0, self.i + 1), 0)
        else:
            return (getID(self.f, self.s, self.t, self.o + 2, self.i), 0)


# 야구 선수 정보를 나타내는 클래스
class Player:
    """
    야구 선수를 나타내는 클래스입니다.
    """

    def __init__(self, playerID, name, first, second, third, bb, homerun, outs,
                 double):
        """
        :param playerID: int. 선수의 고유 ID.
        :param name: string. 선수의 이름.
        :param first: float. 단타 확률.
        :param second: float. 2루타 확률.
        :param third: float. 3루타 확률.
        :param bb: float. 볼넷 확률.
        :param homerun: float. 홈런 확률.
        :param outs: float. 아웃 확률.
        :param double: float. 병살 확률.
        """
        self.id = playerID
        self.name = name
        self.first = first
        self.second = second
        self.third = third
        self.double = double
        self.bb = bb
        self.outs = outs
        self.homerun = homerun

    def transitionMatrixSimple(self):
        """
        이 선수에 대한 전이 행렬을 계산합니다.
        :return: numpy (217, 217) 배열. 이 선수의 전이 행렬.
        """
        p = np.zeros((5, 217, 217))
        p[0][216][216] = 1

        for i in range(216):
            currState = State(i)
            nextState, runs = currState.walk()
            p[runs][i][nextState] += self.bb
            nextState, runs = currState.single()
            p[runs][i][nextState] += self.first
            nextState, runs = currState.double()
            p[runs][i][nextState] += self.second
            nextState, runs = currState.triple()
            p[runs][i][nextState] += self.third
            nextState, runs = currState.homeRun()
            p[runs][i][nextState] += self.homerun
            nextState, runs = currState.out()
            p[runs][i][nextState] += self.outs
            nextState, runs = currState.doublePlay()
            p[runs][i][nextState] += self.double
        return p


def expectedRuns(lineup):
    """
    주어진 야구 라인업의 예상 득점 분포를 계산합니다.
    :param lineup: [Batter]. 라인업에 포함된 9명의 타자 리스트.
    :return: np.array. 21개의 요소를 포함하는 배열. i번째 요소는 라인업이 i 득점할 확률을 나타냅니다.
    """
    transitionsMatrices = list(
        map(lambda Batter: Batter.transitionMatrixSimple(), lineup))
    return simulateMarkovChain(transitionsMatrices)[:, 216]


def simulateMarkovChain(transitionMatrices):
    """
    야구 게임을 나타내는 마르코프 체인의 거의 정적 상태 분포를 찾습니다.
    :param transitionMatrices: [numpy array]. 라인업에 포함된 타자에 대한 9개의 (217x217) 전이 행렬 리스트.
    :return: numpy 21x217 배열. 배열의 i번째 행은 i 득점이 된 상태를 나타냅니다.
    """
    u = np.zeros((21, 217))
    u[0][0] = 1
    iterations = 0
    batter = 0
    while sum(u)[216] < 0.999 and iterations < 2000:
        p = transitionMatrices[batter]
        next_u = np.zeros((21, 217))
        for i in range(21):
            for j in range(5):
                if i - j >= 0:
                    next_u[i] += u[i - j] @ p[j]
        u = next_u
        batter = (batter + 1) % 9
        iterations += 1
    return u


def teamExpectedRuns(teamName, opponent_team_name, starter_lineup_list,
                     relief_lineup_list, starter_data, starter_name,
                     starter_num):
    """
    주어진 팀의 예상 득점을 계산하고 결과를 출력합니다.
    :param teamName: 팀 이름.
    :param starter_lineup_list: 선발 투수 라인업 리스트.
    :param relief_lineup_list: 구원 투수 라인업 리스트.
    :param starter_data: 선발 투수 데이터.
    :param starter_num: 선발 투수 번호.
    :param opponent_name: 상대팀 이름
    """
    print('\n팀: ' + teamName + '\n')
    print('상대팀: ' + opponent_team_name + '\n')
    print('상대 선발 투수: ' + starter_name + '\n')
    print('라인업: ' +
          str(list(map(lambda Batter: Batter.name, starter_lineup_list[0]))) +
          '\n')

    starter_num = list(
        pybaseball.playerid_reverse_lookup(
            [starter_num], key_type='mlbam')['key_fangraphs'])[0]
    try:
        inning = starter_data[starter_data['IDfg'] ==
                              starter_num].loc[:, 'Start-IP']
        game_started = starter_data[starter_data['IDfg'] ==
                                    starter_num].loc[:, 'GS']
        avg_inning = float(inning / game_started)
        if avg_inning < 5.0:
            avg_inning = 5.0
    except:
        avg_inning = 5.0

    # 선발 투수 득점 계산
    u = expectedRuns(starter_lineup_list[0])
    starter_expRuns = 0
    if sum(u) < 0.7:
        print('게임 종료 확률이 낮아 예상 실점을 계산할 수 없습니다.')
        u = (1 / sum(u)) * u

        for i in range(21):
            starter_expRuns += i * u[i]

        avg_inning = 9 * (4 / starter_expRuns)
    else:
        for i in range(21):
            starter_expRuns += i * u[i]
        if (avg_inning / 9) * starter_expRuns > 4:
            avg_inning = 9 * (4 / starter_expRuns)

    # 불펜 투수 득점 계산

    relief_exp_runs_list = []
    for relief_lineup in relief_lineup_list:
        exp = expectedRemainingRuns(relief_lineup, 0,
                                    State(getID(0, 0, 0, 0, 1)))
        relief_exp_runs_list.append(exp)

    relief_expRuns = sum(relief_exp_runs_list) / len(relief_exp_runs_list)
    total_expRuns = (avg_inning / 9) * starter_expRuns + (
        (9 - avg_inning) / 9) * relief_expRuns

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(u)), u, color='blue')
    ax.set_xlabel('Runs Scored')
    ax.set_ylabel('Probability')
    ax.set_title(f'{teamName}의 선발 상대 예상 득점 분포')
    ax.legend()

    # 예상 득점 텍스트 추가
    anchored_text = AnchoredText(
        f'상대 팀: {opponent_team_name}\nStarter Expected Runs: {(avg_inning / 9) * starter_expRuns:.2f}\nRelief Expected Runs: {((9 - avg_inning) / 9) * relief_expRuns:.2f}\nTotal Expected Runs: {total_expRuns:.2f}',
        loc='upper right',
        prop=dict(size=10))
    ax.add_artist(anchored_text)

    plt.savefig(f'img/{teamName}.png')

    print('게임 종료 확률: ' + str(sum(u)) + '\n')
    print('\n선발 투수 평균 이닝 수: ' + str(avg_inning) + '\n')
    print('선발 투수에 의한 예상 실점: ' + str((avg_inning / 9) * starter_expRuns) + '\n')
    print('선발 투수에 의한 예상 실점(9이닝 당): ' + str(starter_expRuns) + '\n')

    print('각 점수에 대한 확률:')
    for i in range(21):
        print(str(i) + ': ' + str(u[i]))

    print('\n구원 투수 예상 실점: ' + str(((9 - avg_inning) / 9) * relief_expRuns) +
          '\n')

    print('\n총 예상 실점:' + str(total_expRuns) + '\n')

    return total_expRuns


def expectedRemainingRuns(lineup, batterUp, startState):
    """
    게임의 특정 지점에서 팀이 득점할 예상 점수를 계산합니다.
    :param lineup: 9명의 타자 리스트.
    :param batterUp: 타석에 있는 타자의 인덱스 (0-8).
    :param startState: 현재 게임의 상태.
    :return: 주어진 상태에서 팀이 득점할 예상 점수.
    """
    transitionsMatrices = list(
        map(lambda Batter: Batter.transitionMatrixSimple(), lineup))
    u = np.zeros((21, 217))
    u[0][startState.id] = 1
    iterations = 0
    batter = batterUp
    while sum(u)[216] < 0.999 and iterations < 2000:
        p = transitionsMatrices[batter]
        next_u = np.zeros((21, 217))
        for i in range(21):
            for j in range(5):
                if i - j >= 0:
                    next_u[i] += u[i - j] @ p[j]
        u = next_u
        batter = (batter + 1) % 9
        iterations += 1
    u = u[:, 216]
    expRuns = 0
    if sum(u) < 0.7:
        # u = (1/sum(u))*u
        expRuns = 9
    else:
        for i in range(21):
            expRuns += i * u[i]
        if expRuns > 9:
            expRuns = 9
    return expRuns


def pitcher_batter_aug(pitcher_data, batter_data, fielding_data, home_team,
                       away_team):
    """
    투수와 타자 데이터를 결합하여 필요한 데이터를 생성합니다.
    :param pitcher_data: 투수 데이터.
    :param batter_data: 타자 데이터.
    :param home_team: 홈팀 경기장 정보.
    :return: 결합된 데이터 리스트.
    """
    batter_pitcher_list = []
    for pitcher_num in list(pitcher_data['pitcher_key_mlbam']):
        batter_pitcher = pd.DataFrame()
        for player_num in list(batter_data['batter_key_mlbam']):
            row = pd.DataFrame({
                'home_team': [home_team],
                'away_team': [away_team],
                'pitcher_key_mlbam': [pitcher_num],
                'batter_key_mlbam': [player_num],
            })
            batter_pitcher = pd.concat([batter_pitcher, row],
                                       ignore_index=True)
        batter_pitcher = pd.merge(batter_pitcher, pitcher_data)
        batter_pitcher = pd.merge(batter_pitcher, batter_data)
        batter_pitcher = pd.merge(batter_pitcher, fielding_data)

        park_factors = {
            'COL': 112,
            'BOS': 107,
            'KC': 105,
            'CIN': 104,
            'TEX': 102,
            'WSH': 102,
            'LAA': 101,
            'STL': 101,
            'HOU': 101,
            'ATL': 101,
            'PHI': 101,
            'MIN': 101,
            'TOR': 100,
            'AZ': 100,
            'CHC': 100,
            'PIT': 100,
            'MIA': 100,
            'CWS': 99,
            'LAD': 99,
            'MIL': 99,
            'NYY': 99,
            'BAL': 98,
            'DET': 98,
            'OAK': 97,
            'TB': 97,
            'CLE': 96,
            'SF': 96,
            'SD': 96,
            'NYM': 95,
            'SEA': 92
        }
        batter_pitcher['park_factor'] = park_factors[home_team]

        # 필요한 열 선택
        columns_to_select = [
            'home_team', 'away_team', 'batter_key_mlbam', 'pitcher_key_mlbam',
            'IDfg', '(P) IDfg', 'key_bbref', '(P) key_bbref', 'Name',
            '(P) Name', 'Team', '(P) Team', 'bat_split', 'GB/FB', 'LD%', 'GB%',
            'FB%', 'IFFB%', 'HR/FB', 'Spd', 'BsR', 'wFB/C', 'wSL/C', 'wCT/C',
            'wCB/C', 'wCH/C', 'wSF/C', 'O-Swing%', 'Z-Swing%', 'Swing%',
            'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%',
            'SwStr%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%',
            'wFA/C (sc)', 'wFC/C (sc)', 'wFS/C (sc)', 'wFO/C (sc)',
            'wSI/C (sc)', 'wSL/C (sc)', 'wCU/C (sc)', 'wKC/C (sc)',
            'wCH/C (sc)', 'O-Swing% (sc)', 'Z-Swing% (sc)', 'Swing% (sc)',
            'O-Contact% (sc)', 'Z-Contact% (sc)', 'Contact% (sc)',
            'Zone% (sc)', 'LD+%', 'GB%+', 'FB%+', 'HR/FB%+', 'Pull%+',
            'Cent%+', 'Oppo%+', 'Soft%+', 'Med%+', 'Hard%+', 'EV', 'LA',
            'Barrel%', 'maxEV', 'HardHit%', 'CStr%', 'CSW%', 'wCH/C (pi)',
            'wCU/C (pi)', 'wFA/C (pi)', 'wFC/C (pi)', 'wFS/C (pi)',
            'wSI/C (pi)', 'wSL/C (pi)', 'O-Swing% (pi)', 'Z-Swing% (pi)',
            'Swing% (pi)', 'O-Contact% (pi)', 'Z-Contact% (pi)',
            'Contact% (pi)', 'Zone% (pi)', 'Pace', 'UBR',
            'anglesweetspotpercent', 'ev50', 'fbld', 'max_distance',
            'avg_distance', 'avg_hr_distance', 'ev95percent',
            '(P) pitch_split', '(P) GB/FB', '(P) LD%', '(P) GB%', '(P) FB%',
            '(P) IFFB%', '(P) HR/FB', '(P) FB% 2', '(P) FBv', '(P) SL%',
            '(P) SLv', '(P) CT%', '(P) CTv', '(P) CB%', '(P) CBv', '(P) CH%',
            '(P) CHv', '(P) SF%', '(P) SFv', '(P) wFB/C', '(P) wSL/C',
            '(P) wCT/C', '(P) wCB/C', '(P) wCH/C', '(P) wSF/C', '(P) O-Swing%',
            '(P) Z-Swing%', '(P) Swing%', '(P) O-Contact%', '(P) Z-Contact%',
            '(P) Contact%', '(P) Zone%', '(P) F-Strike%', '(P) SwStr%',
            '(P) FA% (sc)', '(P) FC% (sc)', '(P) FS% (sc)', '(P) FO% (sc)',
            '(P) SI% (sc)', '(P) SL% (sc)', '(P) CU% (sc)', '(P) KC% (sc)',
            '(P) CH% (sc)', '(P) vFA (sc)', '(P) vFC (sc)', '(P) vFS (sc)',
            '(P) vFO (sc)', '(P) vSI (sc)', '(P) vSL (sc)', '(P) vCU (sc)',
            '(P) vKC (sc)', '(P) vCH (sc)', '(P) vKN (sc)', '(P) FA-X (sc)',
            '(P) FC-X (sc)', '(P) FS-X (sc)', '(P) FO-X (sc)', '(P) SI-X (sc)',
            '(P) SL-X (sc)', '(P) CU-X (sc)', '(P) KC-X (sc)', '(P) CH-X (sc)',
            '(P) FA-Z (sc)', '(P) FC-Z (sc)', '(P) FS-Z (sc)', '(P) FO-Z (sc)',
            '(P) SI-Z (sc)', '(P) SL-Z (sc)', '(P) CU-Z (sc)', '(P) KC-Z (sc)',
            '(P) CH-Z (sc)', '(P) wFA/C (sc)', '(P) wFC/C (sc)',
            '(P) wFS/C (sc)', '(P) wFO/C (sc)', '(P) wSI/C (sc)',
            '(P) wSL/C (sc)', '(P) wCU/C (sc)', '(P) wKC/C (sc)',
            '(P) wCH/C (sc)', '(P) O-Swing% (sc)', '(P) Z-Swing% (sc)',
            '(P) Swing% (sc)', '(P) O-Contact% (sc)', '(P) Z-Contact% (sc)',
            '(P) Contact% (sc)', '(P) Zone% (sc)', '(P) LD%+', '(P) GB%+',
            '(P) FB%+', '(P) HR/FB%+', '(P) Pull%+', '(P) Cent%+',
            '(P) Oppo%+', '(P) Soft%+', '(P) Med%+', '(P) Hard%+', '(P) EV',
            '(P) LA', '(P) Barrel%', '(P) maxEV', '(P) HardHit%', '(P) CStr%',
            '(P) CSW%', '(P) CH% (pi)', '(P) CU% (pi)', '(P) FA% (pi)',
            '(P) FC% (pi)', '(P) FS% (pi)', '(P) SI% (pi)', '(P) SL% (pi)',
            '(P) vCH (pi)', '(P) vCU (pi)', '(P) vFA (pi)', '(P) vFC (pi)',
            '(P) vFS (pi)', '(P) vSI (pi)', '(P) vSL (pi)', '(P) CH-X (pi)',
            '(P) CU-X (pi)', '(P) FA-X (pi)', '(P) FC-X (pi)', '(P) FS-X (pi)',
            '(P) SI-X (pi)', '(P) SL-X (pi)', '(P) CH-Z (pi)', '(P) CU-Z (pi)',
            '(P) FA-Z (pi)', '(P) FC-Z (pi)', '(P) FS-Z (pi)', '(P) SI-Z (pi)',
            '(P) SL-Z (pi)', '(P) wCH/C (pi)', '(P) wCU/C (pi)',
            '(P) wFA/C (pi)', '(P) wFC/C (pi)', '(P) wFS/C (pi)',
            '(P) wSI/C (pi)', '(P) wSL/C (pi)', '(P) O-Swing% (pi)',
            '(P) Z-Swing% (pi)', '(P) Swing% (pi)', '(P) O-Contact% (pi)',
            '(P) Z-Contact% (pi)', '(P) Contact% (pi)', '(P) Zone% (pi)',
            '(P) Pace', '(P) anglesweetspotpercent', '(P) ev50', '(P) fbld',
            '(P) max_distance', '(P) avg_distance', '(P) avg_hr_distance',
            '(P) ev95percent', 'ARM/G', 'DPR/G', 'RngR/G', 'ErrR/G', 'UZR/G',
            'Def/G', 'FRM/G', 'OAA/G', 'Range/G', 'park_factor'
        ]

        batter_pitcher = batter_pitcher[columns_to_select]
        batter_pitcher_list.append(batter_pitcher)

    return batter_pitcher_list


def make_prob_lineup(data_list, load_scaler, model):
    """
    타자와 투수 데이터를 사용하여 확률적 라인업을 생성합니다.
    :param data_list: 결합된 데이터 리스트.
    :param load_scaler: 스케일러 객체.
    :param model: 예측 모델.
    :return: 확률적 라인업 리스트.
    """
    lineup_list = []
    for data in data_list:
        x = data.iloc[:, 12:].values
        x = load_scaler.transform(x)
        y_predict = model.predict(x)

        lineup = []
        playerIDs = list(data.iloc[:, 2])
        playerNames = list(data.iloc[:, 8])

        for i in range(9):
            playerID = playerIDs[i]
            name = playerNames[i]
            first = y_predict[i][4]
            second = y_predict[i][0]
            third = y_predict[i][5]
            double = y_predict[i][1]
            bb = y_predict[i][6]
            outs = y_predict[i][3]
            homerun = y_predict[i][2]
            lineup.append(
                Player(playerID, name, first, second, third, bb, homerun, outs,
                       double))

        lineup_list.append(lineup)

    return lineup_list


def today_lineup(bat_recode, pitch_recode, home_fielding_recode,
                 away_fielding_recode, year, date):
    """
    오늘의 경기 라인업을 가져오고 예상 득점을 계산합니다.
    :param bat_recode: 타자 기록 데이터.
    :param pitch_recode: 투수 기록 데이터.
    :param year: 년도.
    :param date: 날짜 (YYYY-MM-DD).
    """

    url = f'https://www.mlb.com/starting-lineups/{date}'

    response = requests.get(url)
    dom = BeautifulSoup(response.content, 'html.parser')

    game_data = dom.find_all(attrs={'class': 'starting-lineups__matchup'})

    results = {}
    j = 0
    game_num = 0
    for game in game_data:
        try:
            match = game.find(
                attrs={
                    'class':
                    'starting-lineups__teams starting-lineups__teams--xs starting-lineups__teams--md starting-lineups__teams--lg'
                })
            start_pitcher_data = game.find_all(
                attrs={'class': 'starting-lineups__pitcher--link'})
            pitcher_split_data = game.find_all(
                attrs={'class': 'starting-lineups__pitcher-pitch-hand'})

            lineup_data = match.text.split('\n')

            home_team = lineup_data[6].split(' ')[-2]
            away_team = lineup_data[3].split(' ')[-2]

            home_start_pitcher = start_pitcher_data[3].text
            away_start_pitcher = start_pitcher_data[1].text

            home_start_pitcher_num = int(
                list(start_pitcher_data[3].attrs.values())[1][-6:])
            away_start_pitcher_num = int(
                list(start_pitcher_data[1].attrs.values())[1][-6:])

            home_start_pitcher_split = preprocessor.split_to_num(
                pitcher_split_data[1].text.replace(' ', '')[1])
            away_start_pitcher_split = preprocessor.split_to_num(
                pitcher_split_data[0].text.replace(' ', '')[1])

            if home_start_pitcher_num in list(
                    pitch_recode['pitcher_key_mlbam']):
                home_start_pitcher_data = pitch_recode[
                    pitch_recode['pitcher_key_mlbam'] ==
                    home_start_pitcher_num]
            else:
                print('해당 투수 정보 없음')
                # 모든 열의 평균 계산
                numeric_means = pitch_recode.select_dtypes(
                    include='number').mean()

                # 비숫자형 열의 첫 번째 값 가져오기
                non_numeric_data = pitch_recode.select_dtypes(
                    exclude='number').iloc[0]

                # 평균값과 비숫자형 데이터를 결합
                mean_data = pd.concat([numeric_means, non_numeric_data])

                # 평균값을 데이터프레임으로 변환
                home_start_pitcher_data = pd.DataFrame(mean_data).transpose()
                home_start_pitcher_data[
                    'pitcher_key_mlbam'] = home_start_pitcher_num
                home_start_pitcher_data['(P) Name'] = home_start_pitcher
                home_start_pitcher_data[
                    '(P) pitch_split'] = home_start_pitcher_split

            if away_start_pitcher_num in list(
                    pitch_recode['pitcher_key_mlbam']):
                away_start_pitcher_data = pitch_recode[
                    pitch_recode['pitcher_key_mlbam'] ==
                    away_start_pitcher_num]
            else:
                print('해당 투수 정보 없음')
                # 모든 열의 평균 계산
                numeric_means = pitch_recode.select_dtypes(
                    include='number').mean()

                # 비숫자형 열의 첫 번째 값 가져오기
                non_numeric_data = pitch_recode.select_dtypes(
                    exclude='number').iloc[0]

                # 평균값과 비숫자형 데이터를 결합
                mean_data = pd.concat([numeric_means, non_numeric_data])

                # 평균값을 데이터프레임으로 변환
                away_start_pitcher_data = pd.DataFrame(mean_data).transpose()
                away_start_pitcher_data[
                    'pitcher_key_mlbam'] = away_start_pitcher_num
                away_start_pitcher_data['(P) Name'] = away_start_pitcher
                away_start_pitcher_data[
                    '(P) pitch_split'] = away_start_pitcher_split

            home_batters_data = pd.DataFrame()
            away_batters_data = pd.DataFrame()

            for i in range(9):
                home_batter = lineup_data[21 + i].split(' (')[0]
                home_position = lineup_data[21 + i].split(' (')[1].split(' ')[1]
                home_split = preprocessor.split_to_num(
                    lineup_data[21 + i].split(' (')[1].split(' ')[0][0])
                home_batter_num = int(
                    list(
                        match.find_all(
                            attrs={'class': 'starting-lineups__player--link'})[
                                i + 9].attrs.values())[1][-6:])

                if home_batter_num in list(bat_recode['batter_key_mlbam']):
                    batter_data = bat_recode[bat_recode['batter_key_mlbam'] ==
                                             home_batter_num]
                    home_batters_data = pd.concat(
                        [home_batters_data, batter_data], ignore_index=True)

                else:
                    # 모든 열의 평균 계산
                    numeric_means = bat_recode.select_dtypes(
                        include='number').mean()

                    # 비숫자형 열의 첫 번째 값 가져오기
                    non_numeric_data = bat_recode.select_dtypes(
                        exclude='number').iloc[0]

                    # 평균값과 비숫자형 데이터를 결합
                    mean_data = pd.concat([numeric_means, non_numeric_data])

                    # 평균값을 데이터프레임으로 변환
                    batter_data = pd.DataFrame(mean_data).transpose()
                    batter_data['batter_key_mlbam'] = home_batter_num
                    batter_data['Name'] = home_batter
                    batter_data['bat_split'] = home_split
                    home_batters_data = pd.concat(
                        [home_batters_data, batter_data], ignore_index=True)

                away_batter = lineup_data[10 + i].split(' (')[0]
                away_position = lineup_data[10 +
                                            i].split(' (')[1].split(' ')[1]
                away_split = preprocessor.split_to_num(
                    lineup_data[10 + i].split(' (')[1].split(' ')[0][0])
                away_batter_num = int(
                    list(
                        match.find_all(
                            attrs={'class': 'starting-lineups__player--link'})
                        [i].attrs.values())[1][-6:])

                if away_batter_num in list(bat_recode['batter_key_mlbam']):
                    batter_data = bat_recode[bat_recode['batter_key_mlbam'] ==
                                             away_batter_num]
                    away_batters_data = pd.concat(
                        [away_batters_data, batter_data], ignore_index=True)

                else:
                    # 모든 열의 평균 계산
                    numeric_means = bat_recode.select_dtypes(
                        include='number').mean()

                    # 비숫자형 열의 첫 번째 값 가져오기
                    non_numeric_data = bat_recode.select_dtypes(
                        exclude='number').iloc[0]

                    # 평균값과 비숫자형 데이터를 결합
                    mean_data = pd.concat([numeric_means, non_numeric_data])

                    # 평균값을 데이터프레임으로 변환
                    batter_data = pd.DataFrame(mean_data).transpose()
                    batter_data['batter_key_mlbam'] = away_batter_num
                    batter_data['Name'] = away_batter
                    batter_data['bat_split'] = away_split
                    away_batters_data = pd.concat(
                        [away_batters_data, batter_data], ignore_index=True)

            # 불펜 투수 계산

            home_relief_data = pd.DataFrame()
            away_relief_data = pd.DataFrame()

            starter_data = pybaseball.pitching_stats(year, qual=10)
            starter_data = starter_data[starter_data['GS'] /
                                        starter_data['G'] > 0.5]
            start_pitcher_mlb_nums = list(starter_data['IDfg'])
            start_pitcher_mlb_nums = list(
                pybaseball.playerid_reverse_lookup(
                    start_pitcher_mlb_nums, key_type='fangraphs')['key_mlbam'])

            home_depth_num = depth_num(home_team)
            away_depth_num = depth_num(away_team)

            for pitcher_num in home_depth_num:
                try:
                    if (pitcher_num in list(pitch_recode['pitcher_key_mlbam'])
                        ) and (pitcher_num not in start_pitcher_mlb_nums):
                        pitcher_data = pitch_recode[
                            pitch_recode['pitcher_key_mlbam'] == pitcher_num]
                        home_relief_data = pd.concat(
                            [home_relief_data, pitcher_data],
                            ignore_index=True)
                except:
                    continue

            for pitcher_num in away_depth_num:
                try:
                    if (pitcher_num in list(pitch_recode['pitcher_key_mlbam'])
                        ) and (pitcher_num not in start_pitcher_mlb_nums):
                        pitcher_data = pitch_recode[
                            pitch_recode['pitcher_key_mlbam'] == pitcher_num]
                        away_relief_data = pd.concat(
                            [away_relief_data, pitcher_data],
                            ignore_index=True)
                except:
                    continue

            home_batter_away_starter_list = pitcher_batter_aug(
                away_start_pitcher_data, home_batters_data,
                away_fielding_recode, home_team, away_team)
            home_batter_away_relief_list = pitcher_batter_aug(
                away_relief_data, home_batters_data, away_fielding_recode,
                home_team, away_team)
            away_batter_home_starter_list = pitcher_batter_aug(
                home_start_pitcher_data, away_batters_data,
                home_fielding_recode, home_team, away_team)
            away_batter_home_relief_list = pitcher_batter_aug(
                home_relief_data, away_batters_data, home_fielding_recode,
                home_team, away_team)

            load_scaler = load(open('scaler_2.pkl', 'rb'))

            model_path = 'models/mlb_model.99.keras'
            model = tf.keras.models.load_model(model_path)

            print('라인업 만들기')
            home_batter_away_starter_lineup_list = make_prob_lineup(
                home_batter_away_starter_list, load_scaler, model)
            home_batter_away_relief_lineup_list = make_prob_lineup(
                home_batter_away_relief_list, load_scaler, model)
            away_batter_home_starter_lineup_list = make_prob_lineup(
                away_batter_home_starter_list, load_scaler, model)
            away_batter_home_relief_lineup_list = make_prob_lineup(
                away_batter_home_relief_list, load_scaler, model)

            print('예상 득점 계산 중')

            home_expRuns = teamExpectedRuns(
                home_team, away_team, home_batter_away_starter_lineup_list,
                home_batter_away_relief_lineup_list, starter_data,
                away_start_pitcher, away_start_pitcher_num)
            away_expRuns = teamExpectedRuns(
                away_team, home_team, away_batter_home_starter_lineup_list,
                away_batter_home_relief_lineup_list, starter_data,
                home_start_pitcher, home_start_pitcher_num)

            # 승률 계산
            exp_runs_diff = home_expRuns - away_expRuns
            home_win_prob = logistic_win_prob(exp_runs_diff)
            away_win_prob = 1 - home_win_prob

            print(f'\n{home_team}의 예상 승률: {home_win_prob:.2%}')
            print(f'{away_team}의 예상 승률: {away_win_prob:.2%}')

            results[game_num] = {
                'home_team': home_team,
                'away_team': away_team,
                'home_expRuns': home_expRuns,
                'away_expRuns': away_expRuns,
                'home_win_prob': home_win_prob,
                'away_win_prob': away_win_prob
            }

            game_num += 1
            j += 1
        except Exception as e:
            print('이 경기는 라인업이 뜨지 않았거나 취소되었습니다.')
            print(f"Error for {e}")

    return results


def depth_num(team_short_name):
    team_long_names = {
        'COL': 'rockies',
        'BOS': 'redsox',
        'KC': 'royals',
        'CIN': 'reds',
        'TEX': 'rangers',
        'WSH': 'nationals',
        'LAA': 'angels',
        'STL': 'cardinals',
        'HOU': 'astros',
        'ATL': 'braves',
        'PHI': 'phillies',
        'MIN': 'twins',
        'TOR': 'bluejays',
        'AZ': 'dbacks',
        'CHC': 'cubs',
        'PIT': 'pirates',
        'MIA': 'marlins',
        'CWS': 'whitesox',
        'LAD': 'dodgers',
        'MIL': 'brewers',
        'NYY': 'yankees',
        'BAL': 'orioles',
        'DET': 'tigers',
        'OAK': 'athletics',
        'TB': 'rays',
        'CLE': 'guardians',
        'SF': 'giants',
        'SD': 'padres',
        'NYM': 'mets',
        'SEA': 'mariners'
    }

    team_name = team_long_names[team_short_name]

    url = f'https://www.mlb.com/{team_name}/roster'

    response = requests.get(url)
    dom = BeautifulSoup(response.content, 'html.parser')

    depth_data = dom.find_all(attrs={'class': 'info'})
    player_num_list = []
    for i in range(len(depth_data)):
        player_num = list(list(depth_data[i])[1].attrs.values())[0][-6:]
        player_num = int(player_num)
        player_num_list.append(player_num)

    return player_num_list


def logistic_win_prob(exp_runs_diff, alpha=0.2):
    """
    기대 득점 차이를 사용하여 승률을 계산합니다.
    :param exp_runs_diff: 기대 득점 차이 (홈 팀 - 어웨이 팀).
    :param alpha: 로지스틱 함수의 민감도 파라미터 (기본값: 0.2).
    :return: 홈 팀 승률.
    """
    return 1 / (1 + np.exp(-alpha * exp_runs_diff))


def save_results_as_image(results, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    text = ""
    for link, result in results.items():
        text += f"\n경기 번호: {link}\n"
        text += f"{result['home_team']} vs {result['away_team']}\n"
        text += f"{result['home_team']} 예상 득점: {result['home_expRuns']}\n"
        text += f"{result['away_team']} 예상 득점: {result['away_expRuns']}\n"
        text += f"{result['home_team']} 승률: {result['home_win_prob']:.2%}\n"
        text += f"{result['away_team']} 승률: {result['away_win_prob']:.2%}\n\n"

    anchored_text = AnchoredText(text,
                                 loc='upper left',
                                 prop=dict(size=10),
                                 frameon=True)
    ax.add_artist(anchored_text)

    ax.set_axis_off()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    
    # 시즌 개막일과 오늘 날짜
    start_date = datetime(2024, 3, 28)
    end_date = datetime.now()

    # 날짜 차이 계산
    delta = end_date - start_date

    bat_recode, _ = preprocessor.bat_recode(2024)
    pitch_recode, _ = preprocessor.pitch_recode(2024)
    home_fielding_recode, _ = preprocessor.fielding_recode(2024, 'home')
    away_fielding_recode, _ = preprocessor.fielding_recode(2024, 'away')

    # 각 날짜에 대해 결과 생성 및 저장
    for i in range(delta.days + 1):
        current_date = start_date + timedelta(days=i)
        current_date_str = current_date.strftime('%Y-%m-%d')
        
        # today_lineup 함수 호출
        results = today_lineup(bat_recode, pitch_recode, home_fielding_recode,
                            away_fielding_recode, 2024, current_date_str)
        
        # 결과 이미지를 날짜에 맞게 저장
        save_results_as_image(results, f'img/results_{current_date_str}.png')
