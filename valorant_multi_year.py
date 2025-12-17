"""
VALORANT Champions 多年度数据爬虫
支持2024和2025年数据爬取和对比分析
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from datetime import datetime
import re
import json
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('valorant_crawl_multi_year.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ValorantMultiYearCrawler:
    def __init__(self):
        self.base_url = "https://www.vlr.gg"
        self.session = requests.Session()
        self.setup_headers()
        
        # 赛事ID配置
        self.events = {
            '2024': {
                'event_id': '2097',  # VALORANT Champions 2024
                'name': 'VALORANT Champions 2024',
                'date': '2024-08'
            },
            '2025': {
                'event_id': '2283',  # VALORANT Champions 2025
                'name': 'VALORANT Champions 2025',
                'date': '2025-09'
            }
        }
        
    def setup_headers(self):
        """设置请求头"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        self.session.headers.update(self.headers)
    
    def get_event_matches(self, event_id, max_pages=5):
        """获取赛事的所有比赛"""
        matches = []
        
        for page in range(1, max_pages + 1):
            url = f"{self.base_url}/event/matches/{event_id}/?page={page}"
            logging.info(f"获取比赛列表第 {page} 页: {url}")
            
            try:
                response = self.session.get(url)
                if response.status_code != 200:
                    logging.warning(f"第 {page} 页获取失败: {response.status_code}")
                    break
                
                soup = BeautifulSoup(response.text, 'html.parser')
                match_items = soup.select('a.wf-module-item.match-item')
                
                if not match_items:
                    logging.info(f"第 {page} 页没有找到比赛，停止爬取")
                    break
                
                logging.info(f"第 {page} 页找到 {len(match_items)} 个比赛")
                
                for item in match_items:
                    if item.get('href'):
                        match_url = self.base_url + item['href']
                        match_id = item['href'].split('/')[1]
                        
                        match_info = {
                            'match_id': match_id,
                            'url': match_url,
                            'href': item['href']
                        }
                        
                        team_elem = item.select_one('.match-item-vs-team')
                        if team_elem:
                            match_info['title'] = team_elem.get_text(strip=True)
                        
                        time_elem = item.select_one('.match-item-time')
                        if time_elem:
                            match_info['time'] = time_elem.get_text(strip=True)
                        
                        stage_elem = item.select_one('.match-item-event')
                        if stage_elem:
                            match_info['stage'] = stage_elem.get_text(strip=True)
                        
                        matches.append(match_info)
                
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                logging.error(f"获取第 {page} 页时出错: {str(e)}")
                break
        
        return matches
    
    def extract_player_stats(self, soup):
        """提取选手统计数据（复用原有逻辑）"""
        players_data = []
        
        team_names = []
        team_links = soup.select('.match-header-link-name')
        for link in team_links[:2]:
            team_names.append(link.get_text(strip=True))
        
        logging.info(f"队伍: {team_names}")
        
        stats_games = soup.select('.vm-stats-game')
        logging.info(f"找到 {len(stats_games)} 个统计区域")
        
        map_index = 0
        
        for game_idx, game in enumerate(stats_games):
            map_elem = game.select_one('.map')
            
            if not map_elem:
                continue
            
            map_text = map_elem.get_text(strip=True)
            
            if not map_text or 'All' in map_text or map_text.strip() == '':
                logging.info(f"跳过累积数据区域: {map_text}")
                continue
            
            # 清理地图文本
            map_text = re.sub(r'\s*PICK\s*', '', map_text)  # 移除PICK
            map_text = re.sub(r'\d+:\d+', '', map_text)      # 移除比分
            map_text = re.sub(r'^\d+\s*', '', map_text)       # 移除开头数字
            
            # 提取地图名（改进算法）
            map_name = None
            
            # 方法1: 直接按冒号分割（处理 "Fracture:55" 这种格式）
            if ':' in map_text:
                potential_name = map_text.split(':')[0].strip()
                if potential_name and len(potential_name) > 3 and potential_name[0].isupper():
                    map_name = potential_name
            
            # 方法2: 查找第一个大写字母开头的长词
            if not map_name:
                words = map_text.split()
                for word in words:
                    # 移除标点符号
                    clean_word = re.sub(r'[^a-zA-Z]', '', word)
                    if clean_word and len(clean_word) > 3 and clean_word[0].isupper():
                        map_name = clean_word
                        break
            
            # 方法3: 尝试匹配已知地图名
            if not map_name:
                known_maps = ['Bind', 'Haven', 'Split', 'Ascent', 'Icebox', 'Breeze', 
                             'Fracture', 'Pearl', 'Lotus', 'Sunset', 'Abyss']
                map_text_lower = map_text.lower()
                for known in known_maps:
                    if known.lower() in map_text_lower:
                        map_name = known
                        break
            
            if not map_name:
                logging.warning(f"无法提取地图名: {map_text}")
                continue
            
            map_index += 1
            logging.info(f"处理地图 {map_index}: {map_name}")
            
            tables = game.select('table.wf-table-inset.mod-overview')
            
            for table_idx, table in enumerate(tables):
                team = team_names[table_idx] if table_idx < len(team_names) else f"Team_{table_idx+1}"
                
                headers = []
                header_row = table.select_one('thead tr')
                if header_row:
                    ths = header_row.select('th')
                    for th in ths:
                        header = th.get('title', '').strip()
                        if not header:
                            header = th.get_text(strip=True)
                        headers.append(header)
                
                if not headers or all(h == '' for h in headers):
                    headers = ['Player', 'Agent', 'R', 'ACS', 'K', 'D', 'A', 
                              '+/-', 'KAST', 'ADR', 'HS%', 'FK', 'FD', '+/- FD']
                
                rows = table.select('tbody tr')
                
                for row in rows:
                    player_data = {
                        'match_map_number': map_index,
                        'map_name': map_name,
                        'team': team
                    }
                    
                    cells = row.select('td')
                    
                    for i, cell in enumerate(cells[:len(headers)]):
                        header = headers[i] if i < len(headers) else f'col{i}'
                        
                        if i == 0:
                            player_elem = cell.select_one('.text-of')
                            if player_elem:
                                href = player_elem.get('href', '')
                                if href and '/' in href:
                                    parts = href.split('/')
                                    if len(parts) >= 4:
                                        player_name = parts[3]
                                    else:
                                        player_name = player_elem.get_text(strip=True)
                                else:
                                    player_name = player_elem.get_text(strip=True)
                                
                                player_data['player_name'] = player_name
                            else:
                                player_data['player_name'] = cell.get_text(strip=True)
                        
                        elif i == 1:
                            agent_img = cell.select_one('img')
                            
                            if agent_img:
                                agent_name = agent_img.get('title', '').strip()
                                if not agent_name:
                                    agent_name = agent_img.get('alt', '').strip()
                                player_data['agent'] = agent_name
                            else:
                                player_data['agent'] = cell.get_text(strip=True)
                        
                        else:
                            cell_text = cell.get_text(' ', strip=True)
                            
                            if header in ['D', 'Deaths', 'd']:
                                clean_text = cell_text.replace('/', ' ')
                                values = [v.strip() for v in clean_text.split() if v.strip()]
                                value = values[0] if values else ''
                            else:
                                values = cell_text.split()
                                value = values[0] if values else ''
                            
                            value = value.replace(',', '').replace('%', '').strip()
                            
                            if header:
                                player_data[header] = value
                    
                    if 'player_name' in player_data:
                        players_data.append(player_data)
        
        return players_data

    def extract_map_stats(self, soup):
        """提取地图统计数据"""
        maps_data = []
        
        map_divs = soup.select('.vm-stats-game')
        
        map_idx = 0
        for div in map_divs:
            map_elem = div.select_one('.map')
            if not map_elem:
                continue
            
            map_text = map_elem.get_text(strip=True)
            
            if not map_text or 'All' in map_text:
                continue
            
            # 清理地图文本
            map_text = re.sub(r'\s*PICK\s*', '', map_text)
            map_text = re.sub(r'\d+:\d+', '', map_text)
            map_text = re.sub(r'^\d+\s*', '', map_text)
            
            # 提取地图名（与extract_player_stats保持一致）
            map_name = None
            
            # 方法1: 按冒号分割
            if ':' in map_text:
                potential_name = map_text.split(':')[0].strip()
                if potential_name and len(potential_name) > 3 and potential_name[0].isupper():
                    map_name = potential_name
            
            # 方法2: 查找大写词
            if not map_name:
                words = map_text.split()
                for word in words:
                    clean_word = re.sub(r'[^a-zA-Z]', '', word)
                    if clean_word and len(clean_word) > 3 and clean_word[0].isupper():
                        map_name = clean_word
                        break
            
            # 方法3: 匹配已知地图
            if not map_name:
                known_maps = ['Bind', 'Haven', 'Split', 'Ascent', 'Icebox', 'Breeze', 
                             'Fracture', 'Pearl', 'Lotus', 'Sunset', 'Abyss']
                map_text_lower = map_text.lower()
                for known in known_maps:
                    if known.lower() in map_text_lower:
                        map_name = known
                        break
            
            if not map_name:
                continue
            
            score_divs = div.select('.score')
            if len(score_divs) >= 2:
                map_idx += 1
                map_data = {
                    'map_number': map_idx,
                    'map_name': map_name,
                    'team1_score': score_divs[0].get_text(strip=True),
                    'team2_score': score_divs[1].get_text(strip=True)
                }
                maps_data.append(map_data)
        
        return maps_data

    def extract_match_info(self, soup, match_url):
        """提取比赛基本信息"""
        info = {
            'match_id': match_url.split('/')[-2] if '/' in match_url else '',
            'url': match_url,
            'crawl_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        teams = soup.select('.match-header-link-name')
        if len(teams) >= 2:
            info['team1'] = teams[0].get_text(strip=True)
            info['team2'] = teams[1].get_text(strip=True)
        
        scores = soup.select('.match-header-vs-score-count')
        if len(scores) >= 2:
            info['team1_score'] = scores[0].get_text(strip=True)
            info['team2_score'] = scores[1].get_text(strip=True)
        
        event_elem = soup.select_one('.match-header-event')
        if event_elem:
            info['event'] = event_elem.get_text(strip=True)
            if 'Final' in info['event'] or 'Semifinal' in info['event']:
                info['match_type'] = 'BO5'
            else:
                info['match_type'] = 'BO3'
        
        return info

    def parse_match_details(self, match_url):
        """解析单场比赛详情"""
        logging.info(f"解析比赛: {match_url}")
        
        try:
            response = self.session.get(match_url)
            if response.status_code != 200:
                logging.warning(f"无法访问比赛页面: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            match_data = self.extract_match_info(soup, match_url)
            players_data = self.extract_player_stats(soup)
            maps_data = self.extract_map_stats(soup)
            
            return {
                'match_info': match_data,
                'players_data': players_data,
                'maps_data': maps_data
            }
            
        except Exception as e:
            logging.error(f"解析失败 {match_url}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_data(self, match_data, match_id, year):
        """保存数据到CSV文件（按年度分类）"""
        # 创建目录
        os.makedirs(f'data/{year}', exist_ok=True)
        os.makedirs(f'data/{year}/raw', exist_ok=True)
        os.makedirs(f'data/{year}/processed', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存比赛信息
        if match_data['match_info']:
            match_df = pd.DataFrame([match_data['match_info']])
            match_df['year'] = year  # 添加年份标识
            match_file = f"data/{year}/raw/match_{match_id}_{timestamp}.csv"
            match_df.to_csv(match_file, index=False, encoding='utf-8-sig')
            
            master_match_file = f"data/{year}/processed/all_matches.csv"
            self.append_to_file(match_df, master_match_file, 'matches')
        
        # 2. 保存选手数据
        if match_data['players_data']:
            players_df = pd.DataFrame(match_data['players_data'])
            players_df['match_id'] = match_id
            players_df['year'] = year  # 添加年份标识
            
            players_file = f"data/{year}/raw/players_{match_id}_{timestamp}.csv"
            players_df.to_csv(players_file, index=False, encoding='utf-8-sig')
            logging.info(f"选手数据保存: {players_file}，共 {len(players_df)} 条记录")
            
            master_players_file = f"data/{year}/processed/all_players.csv"
            self.append_to_file(players_df, master_players_file, 'players')
        
        # 3. 保存地图数据
        if match_data['maps_data']:
            maps_df = pd.DataFrame(match_data['maps_data'])
            maps_df['match_id'] = match_id
            maps_df['year'] = year  # 添加年份标识
            
            maps_file = f"data/{year}/raw/maps_{match_id}_{timestamp}.csv"
            maps_df.to_csv(maps_file, index=False, encoding='utf-8-sig')
            
            master_maps_file = f"data/{year}/processed/all_maps.csv"
            self.append_to_file(maps_df, master_maps_file, 'maps')
    
    def append_to_file(self, new_df, file_path, data_type):
        """追加数据到文件"""
        try:
            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path, encoding='utf-8-sig')
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                if 'match_id' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(
                        subset=['match_id'] if data_type == 'matches' 
                        else ['match_id', 'player_name', 'map_name'] if data_type == 'players'
                        else ['match_id', 'map_number']
                    )
                combined_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            else:
                new_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            logging.warning(f"追加到 {file_path} 失败: {str(e)}")
            new_df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    def crawl_year(self, year, max_matches=50):
        """爬取指定年份的数据"""
        if year not in self.events:
            logging.error(f"不支持的年份: {year}")
            return
        
        event_info = self.events[year]
        logging.info("=" * 60)
        logging.info(f"开始爬取 {event_info['name']} 数据")
        logging.info("=" * 60)
        
        matches = self.get_event_matches(event_info['event_id'], max_pages=5)
        
        if not matches:
            logging.error("没有找到比赛")
            return
        
        logging.info(f"共找到 {len(matches)} 场比赛")
        
        successful = 0
        failed = 0
        
        for i, match in enumerate(matches[:max_matches], 1):
            logging.info(f"[{i}/{min(max_matches, len(matches))}] 爬取: {match.get('title', match['match_id'])}")
            
            match_data = self.parse_match_details(match['url'])
            
            if match_data:
                self.save_data(match_data, match['match_id'], year)
                successful += 1
                
                maps_count = len(match_data['maps_data'])
                players_count = len(match_data['players_data'])
                logging.info(f"  成功：{maps_count}张地图，{players_count}条选手记录")
            else:
                failed += 1
                logging.warning(f"  爬取失败")
            
            delay = random.uniform(5, 10)
            logging.info(f"  等待 {delay:.1f} 秒...")
            time.sleep(delay)
        
        logging.info("=" * 60)
        logging.info(f"{year}年数据爬取完成！成功: {successful}, 失败: {failed}")
        logging.info("=" * 60)
        
        self.generate_summary(year)
    
    def generate_summary(self, year):
        """生成数据汇总"""
        try:
            summary = {
                'year': year,
                'crawl_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            files_to_check = [
                f'data/{year}/processed/all_matches.csv',
                f'data/{year}/processed/all_players.csv', 
                f'data/{year}/processed/all_maps.csv'
            ]
            
            for file_path in files_to_check:
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8-sig')
                        file_name = os.path.basename(file_path)
                        summary[f'{file_name}_records'] = len(df)
                        
                        if 'all_players.csv' in file_path:
                            summary['unique_players'] = df['player_name'].nunique()
                            summary['unique_agents'] = df['agent'].nunique()
                            summary['unique_maps'] = df['map_name'].nunique()
                        
                        if 'all_matches.csv' in file_path:
                            summary['unique_matches'] = df['match_id'].nunique()
                            
                    except Exception as e:
                        summary[f'{file_name}_error'] = str(e)
            
            summary_file = f'data/{year}/crawl_summary_{year}.json'
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logging.info(f"汇总报告保存到: {summary_file}")
            logging.info(f"\n{year}年数据汇总:")
            for key, value in summary.items():
                logging.info(f"  {key}: {value}")
            
        except Exception as e:
            logging.error(f"生成汇总失败: {str(e)}")

    def merge_years_data(self):
        """合并多年度数据用于对比分析"""
        logging.info("=" * 60)
        logging.info("合并多年度数据...")
        logging.info("=" * 60)
        
        os.makedirs('data/merged', exist_ok=True)
        
        # 合并选手数据
        all_years_players = []
        for year in ['2024', '2025']:
            file_path = f'data/{year}/processed/all_players.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                all_years_players.append(df)
                logging.info(f"加载{year}年选手数据: {len(df)}条")
        
        if all_years_players:
            merged_players = pd.concat(all_years_players, ignore_index=True)
            merged_players.to_csv('data/merged/all_players_merged.csv', index=False, encoding='utf-8-sig')
            logging.info(f"合并选手数据完成: {len(merged_players)}条")
        
        # 合并比赛数据
        all_years_matches = []
        for year in ['2024', '2025']:
            file_path = f'data/{year}/processed/all_matches.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                all_years_matches.append(df)
        
        if all_years_matches:
            merged_matches = pd.concat(all_years_matches, ignore_index=True)
            merged_matches.to_csv('data/merged/all_matches_merged.csv', index=False, encoding='utf-8-sig')
            logging.info(f"合并比赛数据完成: {len(merged_matches)}条")
        
        # 合并地图数据
        all_years_maps = []
        for year in ['2024', '2025']:
            file_path = f'data/{year}/processed/all_maps.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                all_years_maps.append(df)
        
        if all_years_maps:
            merged_maps = pd.concat(all_years_maps, ignore_index=True)
            merged_maps.to_csv('data/merged/all_maps_merged.csv', index=False, encoding='utf-8-sig')
            logging.info(f"合并地图数据完成: {len(merged_maps)}条")
        
        logging.info("=" * 60)
        logging.info("多年度数据合并完成！")
        logging.info("=" * 60)

if __name__ == "__main__":
    print("VALORANT Champions 多年度数据爬虫")
    print("=" * 50)
    print("1. 仅爬取2024年数据")
    print("2. 仅爬取2025年数据")
    print("3. 爬取2024+2025年数据（推荐）")
    print("4. 合并已有的多年度数据")
    
    choice = input("\n请选择 (1/2/3/4): ").strip()
    
    crawler = ValorantMultiYearCrawler()
    
    if choice == "1":
        max_matches = int(input("要爬取多少场2024年比赛？(建议30-50): ").strip() or "30")
        crawler.crawl_year('2024', max_matches=max_matches)
    elif choice == "2":
        max_matches = int(input("要爬取多少场2025年比赛？(建议30-50): ").strip() or "30")
        crawler.crawl_year('2025', max_matches=max_matches)
    elif choice == "3":
        max_matches_2024 = int(input("要爬取多少场2024年比赛？(建议30-50): ").strip() or "30")
        crawler.crawl_year('2024', max_matches=max_matches_2024)
        
        max_matches_2025 = int(input("要爬取多少场2025年比赛？(建议30-50): ").strip() or "30")
        crawler.crawl_year('2025', max_matches=max_matches_2025)
        
        crawler.merge_years_data()
    elif choice == "4":
        crawler.merge_years_data()
    else:
        print("无效选择")
