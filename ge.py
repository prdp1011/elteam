import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
import random
from datetime import datetime
import time
import logging
import sys
import os

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create timestamp for log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"dream11_generation_{timestamp}.log")

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("dream11_generator")

# Constants for team composition
TEAM_COMPOSITION = {
    'WK': {'min': 1, 'max': 4},
    'BAT': {'min': 3, 'max': 6},
    'AR': {'min': 1, 'max': 4},
    'BOWL': {'min': 3, 'max': 6}
}

MAX_PLAYERS_PER_TEAM = 7
TOTAL_PLAYERS = 11
NUM_TEAMS_TO_GENERATE = 20  # Reduced from 20 to 5 for faster execution
REQUEST_TIMEOUT = 10  # 5 second timeout for all web requests

def get_player_role(soup):
    """Determine player role based on URL or name"""
    try:

        role_element = soup.find("div", string="Role")
        if role_element:
            role = role_element.find_next("div").text.strip().lower()
            logger.debug(f"Found role from webpage: {role}")
            print(f"Found role from webpage: {role}")

            if "wk" in role or "keeper" in role:
                logger.info(f"Determined role: WK")
                return "WK"
            elif "all" in role:
                logger.info(f"Determined role: AR")
                return "AR"
            elif "bowl" in role:
                logger.info(f"Determined role: BOWL")
                return "BOWL"
            elif "bat" in role:
                logger.info(f"Determined role: BAT")
                return "BAT"
        
        # Fallback: determine role from URL
        logger.debug("No role found on webpage, determining from URL")
        player_name = player_url.split('/')[-1].lower()
        name_hash = sum(ord(c) for c in player_name)
        
        # Distribute roles evenly
        role_num = name_hash % 10
        if role_num < 2:
            logger.info(f"Determined role algorithmically: WK")
            return "WK"  # 20% wicket-keepers 
        elif role_num < 5:
            logger.info(f"Determined role algorithmically: BAT")
            return "BAT"  # 30% batsmen
        elif role_num < 8:
            logger.info(f"Determined role algorithmically: BOWL")
            return "BOWL"  # 30% bowlers
        else:
            logger.info(f"Determined role algorithmically: AR")
            return "AR"  # 20% all-rounders
        
    except Exception as e:
        logger.error(f"Error getting player role: {e}")
        # Fallback to algorithm
        player_name = player_url.split('/')[-1].lower()
        name_hash = sum(ord(c) for c in player_name)
        roles = ["BAT", "BOWL", "AR", "WK"]
        role = roles[name_hash % len(roles)]
        logger.info(f"Using fallback role: {role}")
        return role

def get_player_stats(player_url, team_mapping=None):
    """Get player stats with web scraping and fallback to random data"""
    logger.info(f"Getting stats for player: {player_url}")
    start_time = time.time()
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        logger.debug(f"Sending request to: {player_url}")
        response = requests.get(player_url, headers=headers, timeout=REQUEST_TIMEOUT)
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract player name
        name_tag = soup.find("h1", class_="cb-font-40")
        name = name_tag.text.strip() if name_tag else player_url.split('/')[-1].replace('-', ' ').title()
        logger.debug(f"Extracted player name: {name}")
        
        # Get player ID from URL for team mapping
        player_id = player_url.split('/')[-1]
        
        # Extract team information - try various selectors or use mapping
        if team_mapping and player_id in team_mapping:
            team = team_mapping[player_id]
            logger.debug(f"Using team mapping for {name}: {team}")
        else:
            team_info = soup.find("div", class_="cb-col cb-col-40")
            team = team_info.text.strip() if team_info else "Unknown Team"
            logger.debug(f"Extracted team from page for {name}: {team}")
        
        # Get player role
        role = get_player_role(soup)
        logger.debug(f"Got player role: {role}")
        
        # Find performance tables
        tables = soup.find_all('table', class_='table cb-col-100 cb-plyr-thead')
        logger.debug(f"Found {len(tables)} stats tables")
        
        # Initialize with default values based on role (no randomness)
        if role == "BAT":
            batting_avg = 30.0
            strike_rate = 135.0
            economy = 10.0
            wickets = 2.0
            batting_50s = 8.0
            batting_100s = 2.0
            boundary_percent = 50.0
            dot_percent = 20.0
        elif role == "BOWL":
            batting_avg = 15.0
            strike_rate = 110.0
            economy = 7.5
            wickets = 25.0
            batting_50s = 0.0
            batting_100s = 0.0
            boundary_percent = 30.0
            dot_percent = 40.0
        elif role == "AR":
            batting_avg = 25.0
            strike_rate = 130.0
            economy = 8.0
            wickets = 15.0
            batting_50s = 5.0
            batting_100s = 0.0
            boundary_percent = 45.0
            dot_percent = 30.0 
        else:  # WK
            batting_avg = 28.0
            strike_rate = 140.0
            economy = 10.0
            wickets = 0.0
            batting_50s = 6.0
            batting_100s = 1.0
            boundary_percent = 48.0
            dot_percent = 20.0
            
        matches_played = 40  # Default value
        
        # Try to extract real stats from tables
        if tables and len(tables) >= 2:
            logger.debug("Attempting to extract batting and bowling stats from tables")
            try:
                # Extract batting stats
                batting_rows = tables[0].find_all('tr')
                for row in batting_rows:
                    cols = row.find_all('td')
                    if cols and len(cols) > 6 and 'IPL' in row.text:
                        try:
                            # Extract based on IPL table structure:
                            # IPL  M  Inn  Runs  BF  HS  Avg  SR  NO  4s  6s  50  100  200
                            matches_played = int(cols[1].text.strip()) if cols[1].text.strip().isdigit() else matches_played
                            innings = int(cols[2].text.strip()) if cols[2].text.strip().isdigit() else 0
                            runs = int(cols[3].text.strip()) if cols[3].text.strip().isdigit() else 0
                            balls_faced = int(cols[4].text.strip()) if cols[4].text.strip().isdigit() else 0
                            highest_score = cols[5].text.strip()
                            batting_avg = float(cols[6].text.strip()) if cols[6].text.strip() != '-' else 0.0
                            strike_rate = float(cols[7].text.strip()) if cols[7].text.strip() != '-' else 0.0
                            not_outs = int(cols[8].text.strip()) if cols[8].text.strip().isdigit() else 0
                            fours = int(cols[9].text.strip()) if cols[9].text.strip().isdigit() else 0
                            sixes = int(cols[10].text.strip()) if cols[10].text.strip().isdigit() else 0
                            batting_50s = int(cols[11].text.strip()) if cols[11].text.strip().isdigit() else 0
                            batting_100s = int(cols[12].text.strip()) if cols[12].text.strip().isdigit() else 0
                            
                            # Calculate boundary percentage if we have balls faced
                            if balls_faced > 0:
                                boundary_percent = ((fours + sixes) / balls_faced) * 100
                            else:
                                boundary_percent = 0
                                
                            logger.debug(f"Extracted batting stats: M={matches_played}, Inn={innings}, Runs={runs}, BF={balls_faced}, Avg={batting_avg}, SR={strike_rate}, 50s={batting_50s}, 100s={batting_100s}, 4s={fours}, 6s={sixes}")
                            break
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Error extracting batting stats: {e}")
                
                # Extract bowling stats
                bowling_rows = tables[1].find_all('tr')
                for row in bowling_rows:
                    cols = row.find_all('td')
                    if cols and len(cols) > 5 and 'IPL' in row.text:
                        try:
                            # Extract based on IPL table structure:
                            # IPL M  Inn  B  Runs  Wkts  Avg  Econ  SR  BBI  BBM  5w  10w
                            bowling_matches = int(cols[1].text.strip()) if cols[1].text.strip().isdigit() else 0
                            bowling_innings = int(cols[2].text.strip()) if cols[2].text.strip().isdigit() else 0
                            balls_bowled = int(cols[3].text.strip()) if cols[3].text.strip().isdigit() else 0
                            runs_conceded = int(cols[4].text.strip()) if cols[4].text.strip().isdigit() else 0
                            wickets = float(cols[5].text.strip()) if cols[5].text.strip() != '-' else 0.0
                            bowling_avg = float(cols[6].text.strip()) if cols[6].text.strip() != '-' else 0.0
                            economy = float(cols[7].text.strip()) if cols[7].text.strip() != '-' else 0.0
                            bowling_sr = float(cols[8].text.strip()) if cols[8].text.strip() != '-' else 0.0
                            
                            # Calculate dot ball percentage (estimate based on economy)
                            if balls_bowled > 0:
                                # Estimate dot balls based on economy rate
                                # Lower economy typically means more dot balls
                                estimated_scoring_balls = (runs_conceded / balls_bowled) * 6 * (balls_bowled / 6)
                                dot_percent = max(0, min(100, 100 - (estimated_scoring_balls / balls_bowled * 100)))
                            else:
                                dot_percent = 0
                                
                            logger.debug(f"Extracted bowling stats: M={bowling_matches}, Inn={bowling_innings}, B={balls_bowled}, Runs={runs_conceded}, Wkts={wickets}, Avg={bowling_avg}, Econ={economy}, SR={bowling_sr}")
                            break
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Error extracting bowling stats: {e}")
            except Exception as e:
                logger.error(f"Error parsing tables for {name}: {e}")
        
        # Recent form - try to extract from page or use actual data
        form_text = soup.find('div', string='Last 5 Matches')
        recent_form = 0  # Default to 0 instead of random
        
        if form_text:
            form_val_div = form_text.find_next('div')
            if form_val_div:
                form_match_scores = form_val_div.text.strip().split(',')
                recent_form_scores = []
                for score in form_match_scores:
                    try:
                        score = score.strip()
                        if score.isdigit():
                            recent_form_scores.append(int(score))
                        elif score.lower() != 'dnb' and score != '-':
                            # Handle cases like "23*" (not out)
                            clean_score = ''.join(c for c in score if c.isdigit())
                            if clean_score:
                                recent_form_scores.append(int(clean_score))
                    except:
                        pass
                
                if recent_form_scores:
                    recent_form = sum(recent_form_scores) / len(recent_form_scores)
                    logger.debug(f"Extracted recent form: {recent_form} from scores: {recent_form_scores}")
        
        # Calculate consistency factor based on standard deviation of recent performance
        consistency = 0.75  # Default value rather than random
        
        if 'recent_form_scores' in locals() and recent_form_scores:
            # Calculate consistency based on standard deviation - more consistent players have lower std deviation
            if len(recent_form_scores) > 1:
                import numpy as np
                std_dev = np.std(recent_form_scores)
                mean = np.mean(recent_form_scores)
                if mean > 0:
                    # Higher consistency for lower coefficient of variation (std/mean)
                    cv = std_dev / mean
                    consistency = max(0.4, min(0.95, 1 - (cv / 2)))
                    logger.debug(f"Calculated consistency: {consistency} from CV: {cv}")
        
        # Calculate venue adaptability based on actual performance data
        venue_adaptability = 0.8  # Default value rather than random
        
        # Calibrate cost based on role and stats
        base_cost = 8
        
        # Adjust cost based on stats and role
        if role == "BAT":
            stat_factor = (batting_avg * 0.4 + strike_rate * 0.005 + batting_50s * 0.3 + batting_100s * 0.7) / 50
        elif role == "BOWL":
            stat_factor = ((100 - economy * 10) * 0.4 + wickets * 0.3 + dot_percent * 0.01) / 50
        elif role == "AR":
            stat_factor = (batting_avg * 0.3 + strike_rate * 0.003 + (100 - economy * 10) * 0.3 + wickets * 0.2) / 50
        else:  # WK
            stat_factor = (batting_avg * 0.5 + strike_rate * 0.005 + batting_50s * 0.3) / 50
        
        # Clamp stat_factor to reasonable range
        stat_factor = max(0.7, min(1.3, stat_factor))
        
        # Cost calculated based on stats with minimal randomness
        cost = int(round(base_cost * stat_factor * 1.0))
        
        # Ensure cost is within limits to make team selection possible
        cost = max(7, min(9, cost))
        
        # Create player data with enhanced stats
        player_data = {
            "Name": name,
            "Team": team,
            "Role": role,
            "Batting Average": batting_avg,
            "Strike Rate": strike_rate,
            "Bowling Economy": economy,
            "Wickets": wickets,
            "Recent Form": recent_form,
            "Matches Played": matches_played,
            "50s": batting_50s,
            "100s": batting_100s,
            "Boundary %": boundary_percent,
            "Dot Ball %": dot_percent,
            "Consistency": consistency,
            "Venue Adaptability": venue_adaptability,
            "Cost": cost
        }
        print('--------------------------------')
        print(player_data)
        print('--------------------------------')
        logger.info(f"Successfully got stats for {name} ({role}) in {time.time() - start_time:.2f}s")
        return player_data
        
    except Exception as e:
        logger.error(f"Web scraping failed for {player_url}: {e}")
        logger.info("Falling back to default player data")
        
        # Generate player data with reasonable defaults
        player_name = player_url.split('/')[-1].replace('-', ' ').title()
        name_hash = sum(ord(c) for c in player_name.lower())
        
        # Determine role based on name hash
        role_num = name_hash % 10
        if role_num < 2:
            role = "WK"
        elif role_num < 5:
            role = "BAT"
        elif role_num < 8:
            role = "BOWL"
        else:
            role = "AR"
        
        # Determine team
        team = "Team1" if name_hash % 2 == 0 else "Team2"
        
        # Create default stats with more realistic values based on role
        if role == "BAT":
            batting_avg = 30.0
            strike_rate = 135.0
            economy = 10.0
            wickets = 2.0
            boundary_percent = 50.0
            dot_percent = 20.0
            batting_50s = 8.0
            batting_100s = 2.0
        elif role == "BOWL":
            batting_avg = 15.0
            strike_rate = 110.0
            economy = 7.5
            wickets = 25.0
            boundary_percent = 30.0
            dot_percent = 40.0
            batting_50s = 0.0
            batting_100s = 0.0
        elif role == "AR":
            batting_avg = 25.0
            strike_rate = 130.0
            economy = 8.0
            wickets = 15.0
            boundary_percent = 45.0
            dot_percent = 30.0
            batting_50s = 5.0
            batting_100s = 0.0
        else:  # WK
            batting_avg = 28.0
            strike_rate = 140.0
            economy = 10.0
            wickets = 0.0
            boundary_percent = 48.0
            dot_percent = 20.0
            batting_50s = 6.0
            batting_100s = 1.0
        
        player_data = {
            "Name": player_name,
            "Team": team,
            "Role": role,
            "Batting Average": batting_avg,
            "Strike Rate": strike_rate,
            "Bowling Economy": economy,
            "Wickets": wickets,
            "Recent Form": 30.0,  # Average default value
            "Matches Played": 40,   # Default number of matches
            "50s": batting_50s,
            "100s": batting_100s,
            "Boundary %": boundary_percent,
            "Dot Ball %": dot_percent,
            "Consistency": 0.75,    # Default consistency
            "Venue Adaptability": 0.80,  # Default adaptability
            "Cost": 8   # Default middle cost
        }
        logger.info(f"Created default player: {player_name} ({role})")
        return player_data

# Step 1: Fetch Upcoming Match Players
def get_upcoming_match_players():
    logger.info("Fetching upcoming match players...")
    start_time = time.time()
    
    url = "https://www.cricbuzz.com/cricket-series/9237/indian-premier-league-2025/matches"
    headers = {"User-Agent": "Mozilla/5.0"}
    logger.debug(f"Sending request to: {url}")
    response = requests.get(url, headers=headers)
    logger.debug(f"Response status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, "html.parser")
    logger.debug("Parsing response content with BeautifulSoup")
    
    # cb-text-preview
    # cb-text-toss
    # cb-text-inprogress
    match_links = soup.find_all("a", class_="cb-text-inprogress")
    logger.debug(f"Found {len(match_links)} upcoming matches with cb-text-preview class")
    
    if not match_links:
        logger.warning("No upcoming matches found")
        print("No upcoming matches found")
        return [], {}, "Unknown"
    
    # Extract toss winner from the link text
    toss_info = match_links[0].text.strip()
    logger.info(f"Found toss info: {toss_info}")
    toss_winner = "Unknown"
    
    if "opt to" in toss_info:
        # Format is typically "Team Name opt to bowl/bat"
        toss_winner = toss_info.split(" opt to")[0].strip()
        logger.info(f"Extracted toss winner from link: {toss_winner}")
    
    match_url = "https://www.cricbuzz.com" + match_links[0]["href"]
    # https://www.cricbuzz.com/live-cricket-scores/114996/rr-vs-kkr-6th-match-indian-premier-league-2025
    # https://www.cricbuzz.com/cricket-match-squads/114996/rr-vs-kkr-6th-match-indian-premier-league-2025
    match_url = match_url.replace("live-cricket-scores", "cricket-match-squads")
    logger.info(f"Selected match URL: {match_url}")
    
    # print(match_url)
    logger.debug(f"Sending request to match URL: {match_url}")
    response = requests.get(match_url, headers=headers)
    logger.debug(f"Match URL response status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, "html.parser")
    logger.debug("Parsing match page content with BeautifulSoup")
    
    # Extract team names using the specified classes
    team1_element = soup.select_one(".cb-team1 div:last-child")
    team2_element = soup.select_one(".cb-team2 div:last-child") 
    
    team1 = team1_element.text.strip() if team1_element else "Team1"
    team2 = team2_element.text.strip() if team2_element else "Team2"
    
    logger.info(f"Found teams: {team1} vs {team2}")
    print(f"Match: {team1} vs {team2}")
    
    # Ensure toss winner matches one of the teams
    if toss_winner != "Unknown":
        # Check if toss_winner closely matches team1 or team2
        if team1.lower() in toss_winner.lower() or toss_winner.lower() in team1.lower():
            toss_winner = team1
            logger.info(f"Matched toss winner to team1: {toss_winner}")
        elif team2.lower() in toss_winner.lower() or toss_winner.lower() in team2.lower():
            toss_winner = team2
            logger.info(f"Matched toss winner to team2: {toss_winner}")
        else:
            # If no match found, choose one randomly
            logger.warning(f"Toss winner '{toss_winner}' doesn't match any team name. Choosing randomly.")
            toss_winner = random.choice([team1, team2])
            logger.info(f"Randomly selected toss winner: {toss_winner}")
    else:
        # If toss winner is still unknown, pick one of the teams randomly
        toss_winner = random.choice([team1, team2])
        logger.info(f"No clear toss winner found, randomly selected: {toss_winner}")
    
    # Get the first cb-play11-lft-col and cb-play11-rt-col divs
    left = soup.find("div", class_="cb-play11-lft-col")
    right = soup.find("div", class_="cb-play11-rt-col")
    logger.debug(f"Found left squad column: {left is not None}")
    logger.debug(f"Found right squad column: {right is not None}")

    # Extract <a> tags with class "cb-col-100" from both divs
    player_profiles = []
    team_mapping = {}
    
    if left:
        left_links = left.find_all("a", class_="cb-col-100")
        logger.debug(f"Found {len(left_links)} player links in left column")
        player_profiles.extend(left_links)
        # Map these players to team1
        for link in left_links:
            if link.get("href") and "/profiles/" in link["href"]:
                player_id = link["href"].split("/")[-1]
                team_mapping[player_id] = team1
    
    if right:
        right_links = right.find_all("a", class_="cb-col-100")
        logger.debug(f"Found {len(right_links)} player links in right column")
        player_profiles.extend(right_links)
        # Map these players to team2
        for link in right_links:
            if link.get("href") and "/profiles/" in link["href"]:
                player_id = link["href"].split("/")[-1]
                team_mapping[player_id] = team2
    
    logger.debug(f"Total player profiles found: {len(player_profiles)}")
    
    player_urls = []
    for p in player_profiles:
        if p.get("href") and "/profiles/" in p["href"]:
            player_url = "https://www.cricbuzz.com" + p["href"]
            player_urls.append(player_url)
    
    logger.debug(f"Extracted {len(player_urls)} player URLs from profiles")
    
    # Deduplicate and limit to 22 players
    unique_urls = list(set(player_urls))
    logger.debug(f"After deduplication: {len(unique_urls)} unique player URLs")
    
    final_urls = unique_urls[:22]
    logger.info(f"Final player count: {len(final_urls)} (limited to max 22)")
    logger.info(f"Player URLs retrieval completed in {time.time() - start_time:.2f}s")
    
    return final_urls, team_mapping, toss_winner

def generate_dream11_team(df, toss_winner="PBKS"):
    """Generate a balanced Dream11 team using the player data"""
    logger.info(f"Generating team with toss winner: {toss_winner}")
    start_time = time.time()
    
    selected_players = []
    total_points = 100  # Starting budget
    team_composition = TEAM_COMPOSITION.copy()
    
    # Log starting budget and requirements
    logger.info(f"Starting budget: {total_points} points")
    logger.info(f"Team composition requirements: {team_composition}")
    
    # Make a deep copy to avoid modifying the original
    df_copy = df.copy()
    
    # Apply toss winner bonus
    logger.debug(f"Applying toss winner bonus for team: {toss_winner}")
    is_toss_winner = df_copy['Team'].str.contains(toss_winner, case=False, regex=False)
    df_copy['Final_Score'] = df_copy['Predicted Score'] * np.where(is_toss_winner, 1.2, 1.0)
    
    # Add stronger randomness for team variety to ensure unique teams
    logger.debug("Adding randomness to scores for team variety")
    # Use different randomness range each time for more variety
    random_min = random.uniform(0.7, 0.9)
    random_max = random.uniform(1.1, 1.3)
    df_copy['Random_Factor'] = np.random.uniform(random_min, random_max, size=len(df_copy))
    df_copy['Final_Score'] = df_copy['Final_Score'] * df_copy['Random_Factor']
    
    # Introduce occasional role preference to create more diversity
    role_preference = random.choice(['None', 'BAT', 'BOWL', 'AR', 'WK'])
    if role_preference != 'None':
        logger.debug(f"Adding preference for {role_preference} role in this team")
        role_boost = np.where(df_copy['Role'] == role_preference, 
                            random.uniform(1.1, 1.3),
                            1.0)
        df_copy['Final_Score'] = df_copy['Final_Score'] * role_boost
    
    # Sort by score with a small random shuffle component
    if random.random() < 0.3:  # 30% chance to add some randomness to sorting
        logger.debug("Adding randomness to player selection order")
        # Get top half of players by score
        top_half = int(len(df_copy) * 0.5)
        top_players = df_copy.iloc[:top_half].copy()
        rest_players = df_copy.iloc[top_half:].copy()
        
        # Shuffle the top players slightly
        shuffled_indices = np.random.permutation(len(top_players))
        top_players = top_players.iloc[shuffled_indices].reset_index(drop=True)
        
        # Recombine dataframe
        df_copy = pd.concat([top_players, rest_players]).reset_index(drop=True)
    else:
        df_copy = df_copy.sort_values('Final_Score', ascending=False)
    
    # Calculate average player cost
    avg_cost = df_copy['Cost'].mean()
    logger.info(f"Average player cost: {avg_cost:.2f}")
    
    # Calculate expected total cost based on average cost and TOTAL_PLAYERS
    expected_total_cost = avg_cost * TOTAL_PLAYERS
    if expected_total_cost > total_points:
        # Adjust player costs if we can't afford a full team
        adjustment_factor = total_points / expected_total_cost
        logger.warning(f"Expected cost {expected_total_cost:.2f} exceeds budget. Adjusting costs by factor {adjustment_factor:.2f}")
        df_copy['Cost'] = df_copy['Cost'] * adjustment_factor
        
    # First select minimum players from each role
    logger.info("Selecting minimum required players for each role")
    for role in ['WK', 'BAT', 'AR', 'BOWL']:
        min_players = team_composition[role]['min']
        role_players = df_copy[df_copy['Role'] == role]
        logger.debug(f"Need minimum {min_players} {role} players, found {len(role_players)} available")
        
        # Select minimum required players
        for i in range(min_players):
            if not role_players.empty and len(selected_players) < TOTAL_PLAYERS:
                player = role_players.iloc[0].to_dict()
                if player['Cost'] <= total_points:
                    selected_players.append(player)
                    total_points -= player['Cost']
                    logger.debug(f"Selected {player['Name']} ({role}) for {player['Cost']} points, remaining budget: {total_points}")
                    df_copy = df_copy[df_copy['Name'] != player['Name']]  # Remove selected player
                    role_players = df_copy[df_copy['Role'] == role]  # Update role players
                else:
                    logger.warning(f"Cannot afford {player['Name']} ({role}) with cost {player['Cost']} - remaining budget: {total_points}")
                    # Drop this player and try next one
                    role_players = role_players.iloc[1:]
                    df_copy = df_copy[df_copy['Name'] != player['Name']]
            else:
                logger.warning(f"Not enough {role} players available")
    
    # Log current team status
    logger.info(f"Selected {len(selected_players)}/{TOTAL_PLAYERS} players with minimum role requirements")
    logger.info(f"Remaining budget: {total_points} points")
    
    # Fill remaining slots to complete 11 players
    remaining_slots = TOTAL_PLAYERS - len(selected_players)
    logger.info(f"Filling {remaining_slots} remaining slots")
    
    # Calculate how much we can spend per remaining player
    max_per_player = total_points / remaining_slots if remaining_slots > 0 else 0
    logger.info(f"Maximum budget per remaining player: {max_per_player:.2f}")
    
    # If max_per_player is too low, adjust costs further to ensure we get 11 players
    if max_per_player < avg_cost - 1 and remaining_slots > 0:
        # Adjust costs of all remaining players to fit within budget
        logger.warning(f"Budget too tight. Adjusting costs to ensure full team selection.")
        cost_adjustment = max_per_player / avg_cost
        df_copy['Cost'] = df_copy['Cost'] * cost_adjustment
        logger.info(f"Adjusted costs by factor {cost_adjustment:.2f}")
    
    attempts = 0
    max_attempts = 50  # Avoid infinite loops
    
    while remaining_slots > 0 and not df_copy.empty and attempts < max_attempts:
        attempts += 1
        logger.debug(f"Attempt {attempts} to fill remaining slots, {remaining_slots} slots left")
        
        # Look for players that satisfy role maximums
        found_player = False
        for role in ['WK', 'BAT', 'AR', 'BOWL']:
            max_players = team_composition[role]['max']
            current_count = sum(1 for p in selected_players if p['Role'] == role)
            
            if current_count < max_players:
                role_players = df_copy[df_copy['Role'] == role]
                if not role_players.empty:
                    affordable_players = role_players[role_players['Cost'] <= total_points]
                    if not affordable_players.empty:
                        player = affordable_players.iloc[0].to_dict()
                        selected_players.append(player)
                        total_points -= player['Cost']
                        logger.debug(f"Selected {player['Name']} ({role}) for {player['Cost']} points, remaining budget: {total_points}")
                        df_copy = df_copy[df_copy['Name'] != player['Name']]
                        remaining_slots -= 1
                        found_player = True
                break
        
        # If no players match role criteria or are affordable, adjust strategy
        if not found_player:
            logger.warning("No suitable players found. Adjusting strategy...")
            
            # As a last resort, drastically reduce costs to ensure full team
            if attempts > 10 and remaining_slots > 0:
                logger.warning("Drastic cost adjustment to ensure full team")
                df_copy['Cost'] = df_copy['Cost'] * 0.5
            
            # Sort remaining players by cost (ascending) to find cheapest
            df_copy = df_copy.sort_values('Cost', ascending=True)
            
            if not df_copy.empty:
                player = df_copy.iloc[0].to_dict()
                if player['Cost'] <= total_points:
                    selected_players.append(player)
                    total_points -= player['Cost']
                    logger.debug(f"Selected cheapest player {player['Name']} ({player['Role']}) for {player['Cost']} points, remaining budget: {total_points}")
                    df_copy = df_copy[df_copy['Name'] != player['Name']]
                    remaining_slots -= 1
                else:
                    # If we can't afford any players, force budget adjustment
                    logger.error(f"Cannot afford any more players. Cheapest: {player['Cost']}, Budget: {total_points}")
                    df_copy.loc[df_copy.index[0], 'Cost'] = total_points * 0.9  # Set cost to 90% of budget
                    logger.warning(f"Forced cost adjustment for {player['Name']} to {total_points * 0.9}")
            else:
                logger.error("No more players available")
                break
    
    if len(selected_players) < TOTAL_PLAYERS:
        logger.warning(f"Could only select {len(selected_players)}/{TOTAL_PLAYERS} players")
    else:
        logger.info(f"Successfully selected all {TOTAL_PLAYERS} players")
    
    # Log final team composition
    role_count = {}
    for role in ['WK', 'BAT', 'AR', 'BOWL']:
        role_count[role] = sum(1 for p in selected_players if p['Role'] == role)
    logger.info(f"Final team composition: {role_count}")
    
    # Improved captain and vice-captain selection logic with randomized strategies
    logger.info("Selecting captain and vice-captain with role diversity")
    selected_df = pd.DataFrame(selected_players)
    
    if not selected_df.empty:
        # Randomly select a captain selection strategy for more variety
        # Increase chance of non-bowler captain by biasing selection away from pure_score
        captain_strategy = random.choice(['weighted', 'weighted', 'role_based', 'role_based', 'random_top', 'pure_score'])
        logger.debug(f"Using '{captain_strategy}' captain selection strategy")
        
        if captain_strategy == 'weighted':
            # Strategy 1: Role-weighted score (original strategy)
            selected_df = selected_df.sort_values('Final_Score', ascending=False)
            
            # Define role weights to ensure diversity
            # Significantly boost BAT and WK, reduce BOWL weighting
            role_weights = {
                'BAT': 1.3,  # Significantly boost batsmen
                'AR': 1.25,  # Boost all-rounders more (they contribute in both departments)
                'BOWL': 0.85,  # Reduce bowlers to avoid over-selection
                'WK': 1.35    # Boost wicket-keepers the most (they're usually undervalued)
            }
            
            # Apply role weighting to encourage diverse captain selection
            selected_df['Weighted_Score'] = selected_df.apply(
                lambda x: x['Final_Score'] * role_weights.get(x['Role'], 1.0), axis=1
            )
            
            # Add some randomness to prevent always selecting the same roles as captain
            selected_df['Captain_Score'] = selected_df['Weighted_Score'] * np.random.uniform(0.9, 1.1, size=len(selected_df))
            
            # Sort by the weighted captain score
            selected_df = selected_df.sort_values('Captain_Score', ascending=False)
            
        elif captain_strategy == 'pure_score':
            # Strategy 2: Pure score-based (no role weighting)
            selected_df = selected_df.sort_values('Final_Score', ascending=False)
            selected_df['Captain_Score'] = selected_df['Final_Score']
            
        elif captain_strategy == 'role_based':
            # Strategy 3: Prioritize specific roles
            preferred_captain_roles = ['BAT', 'AR', 'WK']  # Added WK to preferred roles
            preferred_candidates = selected_df[selected_df['Role'].isin(preferred_captain_roles)]
            
            if not preferred_candidates.empty:
                preferred_candidates = preferred_candidates.sort_values('Final_Score', ascending=False)
                non_preferred = selected_df[~selected_df['Role'].isin(preferred_captain_roles)]
                non_preferred = non_preferred.sort_values('Final_Score', ascending=False)
                selected_df = pd.concat([preferred_candidates, non_preferred]).reset_index(drop=True)
            else:
                selected_df = selected_df.sort_values('Final_Score', ascending=False)
                
            selected_df['Captain_Score'] = selected_df['Final_Score']
            
        else:  # random_top
            # Strategy 4: Random selection from top players
            selected_df = selected_df.sort_values('Final_Score', ascending=False)
            top_n = min(5, len(selected_df))  # Choose from top 5 players
            if top_n > 0:
                captain_idx = random.randint(0, top_n-1)
                # Move this player to the top
                top_player = selected_df.iloc[captain_idx:captain_idx+1].copy()
                rest_players = pd.concat([selected_df.iloc[:captain_idx], selected_df.iloc[captain_idx+1:]])
                selected_df = pd.concat([top_player, rest_players]).reset_index(drop=True)
            selected_df['Captain_Score'] = selected_df['Final_Score']
            
        # Select captain (top weighted score)
        captain = selected_df.iloc[0]['Name'] if len(selected_df) > 0 else "Unknown"
        captain_role = selected_df.iloc[0]['Role'] if len(selected_df) > 0 else "Unknown"
        
        # For vice-captain, also use different strategies for more variety
        vc_strategy = random.choice(['different_role', 'next_best', 'random_top'])
        logger.debug(f"Using '{vc_strategy}' vice-captain selection strategy")
        
        if vc_strategy == 'different_role':
            # Strategy 1: Prefer a different role than the captain
            vc_candidates = selected_df[selected_df['Role'] != captain_role]
            
            if len(vc_candidates) > 0:
                # Sort by original Final_Score
                vc_candidates = vc_candidates.sort_values('Final_Score', ascending=False)
                vice_captain = vc_candidates.iloc[0]['Name']
            else:
                # If no other role available, fall back to next best
                if len(selected_df) > 1:
                    vice_captain = selected_df.iloc[1]['Name']
                else:
                    vice_captain = "Unknown"
                    
        elif vc_strategy == 'next_best':
            # Strategy 2: Simply pick next highest score
            if len(selected_df) > 1:
                vice_captain = selected_df.iloc[1]['Name']
            else:
                vice_captain = "Unknown"
                
        else:  # random_top
            # Strategy 3: Random from top players (excluding captain)
            if len(selected_df) > 1:
                remaining_df = selected_df.iloc[1:].copy()
                top_n = min(4, len(remaining_df))  # Choose from top 4 remaining players
                if top_n > 0:
                    vc_idx = random.randint(0, top_n-1)
                    vice_captain = remaining_df.iloc[vc_idx]['Name']
                else:
                    vice_captain = remaining_df.iloc[0]['Name']
            else:
                vice_captain = "Unknown"
        
        logger.info(f"Captain: {captain} ({captain_role}), Vice Captain: {vice_captain}")
    else:
        captain = "Unknown"
        vice_captain = "Unknown"
        logger.error("No players to select captain/vice-captain from")
    
    logger.info(f"Team generation completed in {time.time() - start_time:.2f}s with {total_points} points remaining")
    
    return {
        'team': selected_players,
        'captain': captain,
        'vice_captain': vice_captain,
        'total_points': 100 - total_points
    }

def save_to_csv(teams, df):
    """Save teams to a single consolidated log file with nice formatting"""
    try:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        team_logs_dir = "team_logs"
        
        # Create team_logs directory if it doesn't exist
        if not os.path.exists(team_logs_dir):
            os.makedirs(team_logs_dir)
        
        # Create a single consolidated log file for all teams
        consolidated_log_file = os.path.join(team_logs_dir, f"all_teams_{current_time}.txt")
        logger.info(f"Creating consolidated log file: {consolidated_log_file}")
        
        with open(consolidated_log_file, 'w') as log_file:
            log_file.write(f"====== DREAM11 TEAMS GENERATED ON {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ======\n\n")
            
            # Write a summary of all teams first
            log_file.write("TEAMS SUMMARY:\n")
            log_file.write(f"{'TEAM':<5} {'CAPTAIN':<25} {'VICE CAPTAIN':<25} {'POINTS':<10} {'WK':<4} {'BAT':<4} {'AR':<4} {'BOWL':<4}\n")
            log_file.write("-" * 90 + "\n")
            
            for i, team in enumerate(teams):
                player_roles = [p['Role'] for p in team['team']]
                log_file.write(f"{i+1:<5} {team['captain']:<25} {team['vice_captain']:<25} {team['total_points']:<10} "
                               f"{sum(1 for r in player_roles if r == 'WK'):<4} "
                               f"{sum(1 for r in player_roles if r == 'BAT'):<4} "
                               f"{sum(1 for r in player_roles if r == 'AR'):<4} "
                               f"{sum(1 for r in player_roles if r == 'BOWL'):<4}\n")
            
            log_file.write("\n\n")
            
            # Write detailed information for each team
            for i, team in enumerate(teams):
                log_file.write(f"===== DREAM11 TEAM {i+1} =====\n\n")
                log_file.write(f"Captain: {team['captain']}\n")
                log_file.write(f"Vice Captain: {team['vice_captain']}\n")
                log_file.write(f"Total Points Used: {team['total_points']}\n\n")
                
                log_file.write("ROLE DISTRIBUTION:\n")
                log_file.write(f"WK: {sum(1 for p in team['team'] if p['Role'] == 'WK')}\n")
                log_file.write(f"BAT: {sum(1 for p in team['team'] if p['Role'] == 'BAT')}\n")
                log_file.write(f"AR: {sum(1 for p in team['team'] if p['Role'] == 'AR')}\n")
                log_file.write(f"BOWL: {sum(1 for p in team['team'] if p['Role'] == 'BOWL')}\n\n")
                
                log_file.write("PLAYERS:\n")
                log_file.write(f"{'NAME':<25} {'ROLE':<6} {'TEAM':<15} {'PRED SCORE':<12} {'COST':<6}\n")
                log_file.write("-" * 70 + "\n")
                
                # Convert team to DataFrame for sorting
                team_df = pd.DataFrame(team['team'])
                team_df = team_df.sort_values('Final_Score', ascending=False)
                
                for _, player in team_df.iterrows():
                    is_captain = player['Name'] == team['captain']
                    is_vc = player['Name'] == team['vice_captain']
                    
                    # Add (C) for captain and (VC) for vice-captain
                    name_display = f"{player['Name']} (C)" if is_captain else (f"{player['Name']} (VC)" if is_vc else player['Name'])
                    
                    line = f"{name_display:<25} {player['Role']:<6} {player['Team']:<15} {player['Final_Score']:<12.2f} {player['Cost']:<6}\n"
                    log_file.write(line)
                
                log_file.write("\n" + "=" * 80 + "\n\n")
        
        logger.info(f"All teams saved to consolidated log file: {consolidated_log_file}")
        print(f"\nAll teams saved to: {consolidated_log_file}")
        
        # Print preview of first team
        print("\nPREVIEW OF TEAM 1:")
        print(f"Captain: {teams[0]['captain']}")
        print(f"Vice Captain: {teams[0]['vice_captain']}")
        print(f"Total Points Used: {teams[0]['total_points']}")
        print("\nPlayers:")
        team_df = pd.DataFrame(teams[0]['team'])
        team_df = team_df.sort_values('Final_Score', ascending=False)
        team_df = team_df[['Name', 'Role', 'Team', 'Final_Score', 'Cost']]
        team_df.columns = ['Name', 'Role', 'Team', 'Score', 'Cost']
        print(team_df.to_string(index=False))
        
        return consolidated_log_file
    except Exception as e:
        logger.error(f"Error saving team logs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    try:
        logger.info("====== STARTING DREAM11 TEAM GENERATION ======")
        logger.info(f"Timestamp: {datetime.now()}")
        
        start_time = time.time()
        
        # Get player data with web scraping and fallback
        player_urls, team_mapping, toss_winner = get_upcoming_match_players()
        if not player_urls:
            logger.error("No players found. Exiting...")
            return
        
        logger.info(f"Using {toss_winner} as toss winner")
        
        logger.info(f"Fetching stats for {len(player_urls)} players...")
        data = []
        for url in player_urls:
            player_data = get_player_stats(url, team_mapping)
            if player_data:
                data.append(player_data)
        
        df = pd.DataFrame(data)
        
        if df.empty:
            logger.error("No valid player data found. Exiting...")
            return
        
        logger.info(f"Processed {len(df)} players successfully in {time.time() - start_time:.2f} seconds")
        
        # Log player costs and roles
        role_distribution = df['Role'].value_counts().to_dict()
        logger.info(f"Role distribution in data: {role_distribution}")
        logger.info(f"Player cost range: min={df['Cost'].min()}, max={df['Cost'].max()}, avg={df['Cost'].mean():.2f}")
        
        # Feature engineering and prediction model
        logger.info("Training prediction model with enhanced features...")
        
        # Create more sophisticated form score using multiple features
        df["Batting Score"] = (
            df["Batting Average"] * 0.3 +
            df["Strike Rate"] * 0.002 * 100 +  # Scale strike rate impact
            df["50s"] * 2 +                    # Value half-centuries
            df["100s"] * 5 +                   # Value centuries even more
            df["Boundary %"] * 0.2            # Value boundary hitting
        )
        
        df["Bowling Score"] = (
            (10 - df["Bowling Economy"]) * 5 +  # Lower economy is better
            df["Wickets"] * 3 +                # Value wicket-taking ability
            df["Dot Ball %"] * 0.3             # Value dot ball percentage
        )
        
        # Weight score based on role
        df["Role Score"] = np.where(
            df["Role"] == "BAT", df["Batting Score"] * 0.8 + df["Bowling Score"] * 0.2,
            np.where(
                df["Role"] == "BOWL", df["Batting Score"] * 0.2 + df["Bowling Score"] * 0.8,
                np.where(
                    df["Role"] == "AR", df["Batting Score"] * 0.5 + df["Bowling Score"] * 0.5,
                    df["Batting Score"] * 0.7 + df["Bowling Score"] * 0.3  # WK
                )
            )
        )
        
        # Create combined form score considering additional factors
        df["Form Score"] = (
            df["Role Score"] * 0.5 +
            df["Recent Form"] * 0.3 +
            df["Matches Played"].clip(upper=80) / 80 * 10 +  # Experience factor (capped)
            df["Consistency"] * 20 +                         # Value consistency
            df["Venue Adaptability"] * 15                    # Value adaptability
        )
        
        # Prepare features for machine learning
        feature_cols = [
            "Batting Average", "Strike Rate", "Bowling Economy", 
            "Wickets", "Recent Form", "50s", "100s", "Boundary %", 
            "Dot Ball %", "Consistency", "Venue Adaptability", "Matches Played"
        ]
        
        # Create dummy variables for roles
        role_dummies = pd.get_dummies(df["Role"], prefix="Role")
        df = pd.concat([df, role_dummies], axis=1)
        
        # Add role dummy columns to features
        for col in role_dummies.columns:
            feature_cols.append(col)
        
        # Split data into features and target
        X = df[feature_cols].values
        y = df["Form Score"].values
        
        # Train more sophisticated model with hyperparameter tuning
        from sklearn.ensemble import GradientBoostingRegressor
        
        model = GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X, y)
        
        # Generate predictions with the model
        df["Predicted Score"] = model.predict(X)
        
        # Normalize predicted scores to make them more interpretable
        min_score = df["Predicted Score"].min()
        max_score = df["Predicted Score"].max()
        df["Predicted Score"] = 50 + 50 * (df["Predicted Score"] - min_score) / (max_score - min_score)
        
        # Log feature importances
        feature_importances = sorted(zip(feature_cols, model.feature_importances_), 
                                    key=lambda x: x[1], reverse=True)
        logger.info("Feature importances:")
        for feature, importance in feature_importances[:5]:  # Top 5 features
            logger.info(f"  {feature}: {importance:.4f}")
        
        logger.info("Enhanced prediction model training completed")
        
        # Log top predicted players
        top_players = df.nlargest(5, 'Predicted Score')
        logger.info("Top 5 players by predicted score:")
        for _, player in top_players.iterrows():
            logger.info(f"  {player['Name']} ({player['Role']}): {player['Predicted Score']:.2f}")
        
        # Print player data in a more readable format
        print("\n=== PLAYER DATA ===")
        print(f"{'NAME':<25} {'ROLE':<6} {'TEAM':<15} {'PRED SCORE':<12} {'COST':<6}")
        print("-" * 70)
        sorted_players = df.sort_values('Predicted Score', ascending=False)
        for _, player in sorted_players.iterrows():
            print(f"{player['Name']:<25} {player['Role']:<6} {player['Team']:<15} {player['Predicted Score']:<12.2f} {player['Cost']:<6}")
        print("\n")
        
        # Generate teams
        teams = []
        logger.info(f"Generating {NUM_TEAMS_TO_GENERATE} teams...")
        teams = []
        team_hashes = set()  # Keep track of team hashes to ensure uniqueness
        attempts = 0
        max_attempts = NUM_TEAMS_TO_GENERATE * 3  # Limit attempts to avoid infinite loops
        
        while len(teams) < NUM_TEAMS_TO_GENERATE and attempts < max_attempts:
            try:
                attempts += 1
                logger.info(f"Generating team attempt {attempts}...")
                team = generate_dream11_team(df, toss_winner)
                
                # Create a hash of the team based on player names
                team_players = sorted([p['Name'] for p in team['team']])
                team_hash = hash(tuple(team_players))
                
                # Only add the team if it's unique
                if team_hash not in team_hashes:
                    team_hashes.add(team_hash)
                    teams.append(team)
                    
                    logger.info(f"Team {len(teams)} generated (unique):")
                    logger.info(f"Players ({len(team['team'])}): {[p['Name'] for p in team['team']]}")
                    logger.info(f"Captain: {team['captain']}")
                    logger.info(f"Vice Captain: {team['vice_captain']}")
                    logger.info(f"Total Points Used: {team['total_points']}")
                else:
                    logger.info("Generated duplicate team, trying again...")
            except Exception as e:
                logger.error(f"Error generating team: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Successfully generated {len(teams)} unique teams after {attempts} attempts")
        
        # Save teams to a single consolidated log file
        if teams:
            try:
                log_file = save_to_csv(teams, df)
                if log_file:
                    logger.info(f"Teams saved successfully to {log_file}")
                
                logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
                print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
                logger.info(f"Application log file saved to: {log_file}")
                print(f"Application log file saved to: {log_file}")
            except Exception as e:
                logger.error(f"Error saving teams: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"An unhandled error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()