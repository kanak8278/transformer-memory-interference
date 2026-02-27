# Narrative Interference Dataset — Domain Catalog

**Purpose:** Define all candidate domains for generating naturalistic interference trials.
Each domain needs entities with mutable attributes that change over time in a narrative.

**Status:** Draft for review. Each domain will get a detailed spec once approved.

---

## How to read this document

For each domain:
- **Entities**: What we're tracking (= num_keys in the experiment)
- **Sample attributes**: The mutable state that changes (these become what we query for RI/PI)
- **Narrative voice**: How the story reads
- **Scale**: How many entities and updates are natural
- **Constraints to code**: Domain logic needed so random sampling produces valid sequences
- **Status**: Not started / Under review / Approved / Spec complete

We will go through each domain one at a time and flesh out:
1. Complete attribute list with value ranges
2. All domain constraints (what can/can't happen)
3. Narrative templates
4. Entity name pools

---

## GAMES (13 domains)

### 1. Dota 2 Match
- **Entities**: 10 heroes (5v5)
- **Sample attributes**: health, mana, level, gold, net_worth, kills, deaths, assists, last_hits, denies, gpm, xpm, items (6 slots), armor, attack_damage, attack_speed, move_speed, ultimate_cooldown, ability_levels, buyback_status, respawn_timer, tower_damage, hero_damage, position, alive/dead, kill_streak, bkb_duration
- **Narrative voice**: Esports caster / match recap ("At the 22-minute mark, Invoker rotated to the bottom lane with 4.8k gold and picked up a quick double kill, bringing his KDA to 7/2/3.")
- **Scale**: 10 entities, 20-100+ updates per 40-min match
- **Constraints**: Can't buy items with insufficient gold. Can't get kills while dead. Gold/XP only increase (except buyback). Items follow build paths. Levels 1-30 sequential. Respawn timer scales with level.
- **Status**: Not started

### 2. League of Legends Match
- **Entities**: 10 champions (5v5)
- **Sample attributes**: health, mana, level (1-18), gold, kills, deaths, assists, cs, vision_score, wards_placed, wards_destroyed, items (6 slots), attack_damage, ability_power, armor, magic_resist, attack_speed, move_speed, crit_chance, summoner_spell_cooldowns, ultimate_cooldown, dragon_kills, baron_kills, turret_damage, champion_damage, damage_taken, healing_done, shutdown_bounty, alive/dead, respawn_timer
- **Narrative voice**: Similar to Dota but LoL-specific terms ("Jinx completed Infinity Edge at 18 minutes, bringing her crit chance to 40% and attack damage to 287.")
- **Scale**: 10 entities, 20-80+ updates per 30-min match
- **Constraints**: Level cap 18. Item slots 6. Summoner spells have fixed cooldowns. Dragon spawns are sequential (infernal→mountain→ocean→cloud→elder). Baron spawns after 20 min.
- **Status**: Not started

### 3. PUBG / Battle Royale Match
- **Entities**: Up to 100 players or 25 squads
- **Sample attributes**: health, boost_meter, alive/knocked/dead, kills, knocks, assists, damage_dealt, position_x/y, helmet_tier (none/1/2/3), vest_tier, helmet_durability, vest_durability, primary_weapon, secondary_weapon, sidearm, ammo counts, throwables, heal_items, boost_items, backpack_tier, movement_state (prone/crouch/stand/sprint/swim/drive/parachute), in_vehicle, vehicle_type, vehicle_health, vehicle_fuel, distance_from_zone, inside_safe_zone, placement_rank, revives_given, headshot_kills, distance_traveled
- **Narrative voice**: Match replay / observer commentary ("In the third zone collapse, ShadowStrike found a level 3 helmet in Pochinki, swapped his UMP for an M416, and pushed the compound where two squads were fighting.")
- **Scale**: 25-100 entities (but entities die/leave), 10-30 zone phases
- **Constraints**: Players die permanently (or get knocked then die). Zone shrinks on fixed schedule. Loot tiers matter. Armor degrades on hit. Dead players can't act.
- **Status**: Not started

### 4. Fortnite Match
- **Entities**: Up to 100 players
- **Sample attributes**: health, shield (0-100), alive/knocked/dead, kills, assists, wood/brick/metal (0-999 each), weapon slots (5), weapon rarities (common→mythic), ammo types (light/medium/heavy/shells/rockets), structures_built, structures_destroyed, position, inside_storm, storm_damage_per_tick, movement_state (ground/building/gliding/swimming/driving), vehicle_type, crown_holder, damage_dealt, healing_consumed, survival_time, placement
- **Narrative voice**: Faster-paced than PUBG, building emphasis ("NinjaBuild cranked 90s to high ground with 340 wood remaining, tagged the opponent for 86 shield damage with a gold SCAR, then edited down for the elimination.")
- **Scale**: 100 entities initially, shrinking. 10-20 storm phases.
- **Constraints**: Materials deplete when building. Shield before health. Storm damage increases per phase. Building requires materials.
- **Status**: Not started

### 5. Counter-Strike Match
- **Entities**: 10 players (5v5)
- **Sample attributes**: health (0-100), armor (0-100), has_helmet, alive/dead, money (0-16000), kills_total, deaths_total, assists_total, kills_this_round, adr, headshot_pct, primary_weapon (AK47/M4/AWP/etc), secondary_weapon (pistols), flashbangs (0-2), smoke (0-1), HE grenade (0-1), molotov (0-1), has_defuse_kit, has_bomb, mvp_stars, clutch_wins, first_kills, first_deaths, kast_pct, flash_assists, enemies_flashed, utility_damage, damage_this_round, team_side (CT/T), round_score, position_callout, trade_kills, bomb_plants, bomb_defuses, rating
- **Narrative voice**: Round-by-round tactical ("Round 14: s1mple held AWP on mid, got a quick pick on the entry fragger, then rotated B with $7400 in the bank and one smoke remaining.")
- **Scale**: 10 entities, 15-30 rounds per match, multiple events per round
- **Constraints**: Money resets partially each round. Weapons lost on death. Economy system (loss bonus, plant bonus). Side swap at round 15. Equipment costs fixed.
- **Status**: Not started

### 6. Minecraft Survival Server
- **Entities**: Players on multiplayer server (2-50)
- **Sample attributes**: health (0-20 hearts), hunger (0-20), saturation (0-20), xp_level, xp_points, position_x/y/z, dimension (overworld/nether/end), helmet/chestplate/leggings/boots (material types), armor_durability, held_item, held_item_durability, oxygen (underwater), on_fire, potion_effects_active, diamonds_in_inventory, food_items, inventory_slots_used, deaths_total, kills_players, kills_mobs, blocks_mined, distance_walked, gamemode, is_sneaking/sprinting/flying, bed_location_set, enchantments_count, villager_trades, ender_pearls
- **Narrative voice**: Server log / adventure journal ("Steve dug into a diamond vein at Y=-58, pocketing 7 diamonds to bring his total to 23. A creeper snuck up from behind — the explosion dropped him to 4 hearts and shattered his iron chestplate.")
- **Scale**: 5-50 entities, continuous updates
- **Constraints**: Health regenerates with food. Hunger depletes over time. Armor has durability. Tools break. Nether/End have different mob sets. Fall damage depends on height.
- **Status**: Not started

### 7. Civilization Game
- **Entities**: Civilizations (2-12 per game)
- **Sample attributes**: gold, gold_per_turn, science_per_turn, culture_per_turn, faith_per_turn, total_faith, tourism_per_turn, population_total, num_cities, military_strength, num_military_units, num_civilian_units, happiness/amenities, housing, current_era (ancient→future), technologies_researched, civics_completed, current_research, current_civic, government_type, num_wonders, trade_routes, diplomatic_favor, grievances, wars_active, alliances, suzerainties, envoys, great_people_earned, tiles_owned, improved_tiles, strategic_resources (iron/niter/oil/aluminum/uranium), co2_footprint, victory_score, districts_built, food_surplus
- **Narrative voice**: Turn-by-turn history ("Turn 187: Rome completed the Colosseum in their capital, raising culture output to 84/turn. Treasury fell to 230 gold after purchasing a Swordsman in Antium. The war with Egypt enters its 12th turn.")
- **Scale**: 2-12 entities, 100-500 turns
- **Constraints**: Eras are sequential. Technologies have prerequisites. Only one wonder per civ. Resources needed for specific units. Government change costs gold/culture.
- **Status**: Not started

### 8. StarCraft 2 Match
- **Entities**: 2 players (with army composition as attributes)
- **Sample attributes**: minerals (0-99999), vespene (0-99999), food_used, food_cap, worker_count, army_supply, army_value, units_killed, units_lost, structures_killed, structures_lost, apm (actions per minute), current_tech_tier, upgrades_completed, production_building_count, expansion_count, active_army_composition (zealots/stalkers/colossi/etc by count), supply_blocked_time, idle_workers, map_control_pct
- **Narrative voice**: Esports analysis ("At the 8-minute mark, the Protoss player had expanded to a natural with 44 probes, 2 completed gateways, and a Robotics Facility under construction. Mineral bank sat at 350 with 120 vespene.")
- **Scale**: 2 entities with many sub-attributes, 50-200 updates per 15-min game
- **Constraints**: Supply cap 200. Workers gather resources. Buildings have build times. Tech tree is sequential. Can't build units without production buildings.
- **Status**: Not started

### 9. Hearthstone Match
- **Entities**: 2 players + up to 14 board minions
- **Sample attributes**: Per player: health (1-30+), armor (0-999), mana_crystals (0-10), mana_available, cards_in_hand (0-10), cards_in_deck (0-30), fatigue_counter, hero_power_available. Per minion: attack (0-99), health (1-99), damage_taken, taunt, divine_shield, stealth, frozen, poisonous, lifesteal, windfury, silenced, zone_position (1-7), turns_in_play
- **Narrative voice**: Play-by-play ("Turn 7: the Mage played Flamestrike for 4 damage to all minions, clearing three of the Paladin's five recruits. The surviving 4/1 Blessing of Kings target traded up into the 3/6 Water Elemental.")
- **Scale**: 2-16 entities, 20-60 events per 10-turn game
- **Constraints**: Max 7 minions per side. Max 10 mana. Fatigue damage increases by 1 each draw from empty deck. Minions can't attack the turn they're played (unless charge/rush).
- **Status**: Not started

### 10. Poker Tournament
- **Entities**: Players at tables (6-10 per table, 50-1000 in tournament)
- **Sample attributes**: chip_stack, seat_position, table_number, status (active/folded/all_in/eliminated), current_bet, hand_cards (hidden), bounty, hands_played, hands_won, biggest_pot_won, VPIP_pct (voluntary put in pot), PFR_pct (preflop raise), aggression_factor, showdowns_won, bluffs_attempted, position_at_table (button/SB/BB/UTG/etc)
- **Narrative voice**: Poker commentary ("With blinds at 500/1000, Jackson shoved all-in for 12,400 from the cutoff with pocket queens. The big blind tank-called with ace-king suited. The board ran out J-7-2-K-4, and Jackson was eliminated in 23rd place.")
- **Scale**: 6-10 entities per table, 20-200 hands per session
- **Constraints**: Chip stacks must sum to tournament total. Blinds increase on schedule. Can't bet more than stack. Positions rotate. Eliminated players leave.
- **Status**: Not started

### 11. Chess Tournament
- **Entities**: Players (8-128)
- **Sample attributes**: elo_rating, tournament_score (0.0-N.0 in 0.5 increments), wins, draws, losses, color_balance (white minus black games), buchholz_tiebreak, performance_rating, status (active/withdrawn), current_opponent, clock_remaining (seconds), games_with_white, games_with_black, longest_game_moves, shortest_game_moves, average_centipawn_loss
- **Narrative voice**: Tournament report ("After round 5, Grandmaster Petrov leads with 4.5/5 after defeating FM Chen in a 67-move Ruy Lopez endgame. His live rating has climbed to 2687, a gain of 12 points.")
- **Scale**: 8-64 entities, 5-13 rounds
- **Constraints**: Swiss pairing rules (can't play same opponent twice, balance colors). Score only increases by 0, 0.5, or 1 per round. Rating changes follow Elo formula.
- **Status**: Not started

### 12. Among Us Session
- **Entities**: Players (4-15)
- **Sample attributes**: role (crewmate/impostor — hidden), status (alive/dead/ghost), tasks_assigned, tasks_completed, task_completion_pct, color, position_room, sus_level (emergent), votes_received_this_meeting, has_reported_body, emergency_meetings_remaining, kills_committed (impostor only), time_since_last_seen, alibis
- **Narrative voice**: Suspense/mystery recap ("After the body was found in Electrical, Blue reported that they had been in Medbay doing the scan task. Red claimed to have been in Navigation, but Green said they saw Red leaving Electrical just moments earlier. The vote was 4-3 to eject Red.")
- **Scale**: 4-15 entities, 5-15 events per game (kills + meetings)
- **Constraints**: Only impostors can kill. Meetings triggered by body report or emergency button. Ejected players become ghosts. Tasks can only be completed by living crewmates.
- **Status**: Not started

### 13. FIFA / Football Manager Season
- **Entities**: Teams or players across a season
- **Sample attributes**: Per team: points, wins, draws, losses, goals_for, goals_against, goal_difference, position_in_table, form (last 5 results), injuries_count, squad_morale, transfer_budget, wage_bill. Per player: overall_rating, goals, assists, appearances, minutes_played, yellow_cards, red_cards, fitness, form_rating, market_value, contract_years_remaining, injury_status
- **Narrative voice**: Season recap / matchday report ("Matchday 14: City beat United 3-1 at home, climbing to 2nd with 32 points. Striker Rodriguez scored twice, taking his season tally to 11 goals. United's center-back Martinez picked up a red card and will miss the next 3 games.")
- **Scale**: 20 teams or 30+ players, 38 matchdays per season
- **Constraints**: 3 points for win, 1 for draw. Red card = suspension. Injuries have recovery time. Transfer window only open in specific periods. Squad size limits.
- **Status**: Not started

---

## REAL-WORLD MONITORING (11 domains)

### 14. Hospital ICU
- **Entities**: Patients (2-20)
- **Sample attributes**: systolic_bp, diastolic_bp, heart_rate, respiratory_rate, oxygen_saturation, temperature, pain_level (1-10), gcs_score (3-15), urine_output_ml_hr, blood_glucose, potassium, sodium, creatinine, hemoglobin, white_cell_count, platelet_count, lactate, bilirubin, troponin, procalcitonin, inr, ptt, bmi, weight, fluid_balance_ml, ventilator_mode (none/nasal_cannula/high_flow/bipap/ventilator), fio2_pct, peep, tidal_volume, vasopressor (none/norepinephrine/vasopressin/dopamine/epinephrine), vasopressor_dose, sedation_rass (-5 to +4), medication_changes, diet_status, mobility_level, central_line_days, foley_days, wound_status, code_status (full/DNR/comfort)
- **Narrative voice**: Shift handoff / progress note (already proven — "Mrs. Liu's heart rate at the 06:00 check was 92 beats per minute, sinus rhythm.")
- **Scale**: 2-20 entities, 5-50 readings per 24-48hr stay
- **Constraints**: Vitals have physiological ranges. Vasopressor dose changes are gradual. Ventilator settings follow protocols. Electrolytes must be in compatible ranges (e.g., K+ too high or too low is dangerous).
- **Status**: Proven — 9 stories in existing dataset

### 15. Clinical Drug Trial
- **Entities**: Subjects/patients (8-50)
- **Sample attributes**: systolic_bp, diastolic_bp, heart_rate, weight, bmi, hba1c, fasting_glucose, ldl_cholesterol, hdl_cholesterol, triglycerides, alt (liver), ast (liver), creatinine, egfr (kidney function), urine_protein, dosage_level, cohort (treatment_A/B/placebo), compliance_pct, adverse_events (none/nausea/headache/rash/dizziness/fatigue/elevated_liver_enzymes/...), adverse_event_severity (mild/moderate/severe), withdrawal_status (active/withdrawn_adverse/withdrawn_consent/lost_to_followup), concomitant_medications, visit_number, body_temperature, hemoglobin, white_cell_count, platelet_count, procalcitonin, quality_of_life_score, pain_vas_score, sleep_quality_score, exercise_capacity
- **Narrative voice**: Clinical report / DSMB summary ("At the Week 12 visit, Subject PT-0042 presented with HbA1c of 6.8%, down from 9.1% at baseline. Dosage was escalated to 50mg at Week 8. Mild nausea reported but not dose-limiting.")
- **Scale**: 8-50 entities, 8-20 visits over weeks/months
- **Constraints**: Dosage escalation follows protocol. Adverse events may trigger dose reduction. Withdrawal is permanent. Biomarkers have physiological ranges. Compliance affects efficacy.
- **Status**: Not started

### 16. Wildlife Tracking
- **Entities**: Tagged animals (5-30)
- **Sample attributes**: location_zone (named areas), gps_lat, gps_lon, weight_kg, body_condition_score (1-9), health_status (healthy/injured/parasitic/malnourished/lactating), pack_or_group, activity_state (denning/hunting/migrating/resting/foraging), collar_battery_pct, pup_or_calf_count, distance_from_den_km, elevation_m, habitat_type (forest/grassland/riparian/alpine/urban_edge), prey_detected_nearby, interaction_with_tagged_animals, last_capture_date, age_years, reproductive_status (breeding/non-breeding/pregnant/nursing), territory_size_km2, daily_movement_km, overnight_location
- **Narrative voice**: Field report ("March 14: Wolf F-2847 relocated 12 km northeast to the Copper Creek drainage. Weight at last capture was 38.2 kg, down from 41.5 kg. She appears to have separated from the Ridgeline Pack.")
- **Scale**: 5-30 entities, 10-50 observations over weeks/months
- **Constraints**: Animals can't teleport (movement limited per time). Weight changes are gradual. Pack membership changes are rare events. Seasonal behaviors (denning in winter, pup-rearing in spring). Dead animals stop updating.
- **Status**: Not started

### 17. Brewing / Fermentation
- **Entities**: Batches (3-15)
- **Sample attributes**: specific_gravity, ph, temperature_f, yeast_activity (dormant/active/vigorous/flocculating/stalled), stage (mashing/boiling/primary_fermentation/secondary/conditioning/carbonating/kegged/bottled), days_in_stage, dissolved_oxygen_ppb, color_srm, ibu (bitterness), abv_estimated, off_flavors (none/diacetyl/acetaldehyde/sulfur/phenolic/sour), head_space_ml, co2_volumes, pressure_psi, clarity (hazy/clearing/bright), yeast_cell_count_billion, attenuation_pct, original_gravity, final_gravity_target, dry_hop_status (none/added/removed), adjunct_additions, fermentor_type (bucket/carboy/conical/unitank), sample_ph, sample_taste_notes
- **Narrative voice**: Brewer's logbook ("Day 5, Batch IPA-2024-07: Gravity down to 1.024 from 1.062 at pitch. Temperature holding at 66°F. Airlock vigorous, one bubble per second. pH at 4.3. No off-flavors in sample pull.")
- **Scale**: 3-15 entities, 10-30 readings per batch over 2-6 weeks
- **Constraints**: Gravity only decreases during fermentation. Temperature changes affect yeast activity. pH decreases during fermentation. Stages are sequential. Attenuation can't exceed 100%.
- **Status**: Not started

### 18. Air Traffic Control
- **Entities**: Aircraft (10-50 in a sector)
- **Sample attributes**: callsign, altitude_ft, assigned_altitude, speed_knots, assigned_speed, heading_degrees, assigned_heading, vertical_rate_fpm, position_lat, position_lon, squawk_code, aircraft_type, fuel_remaining_lbs, fuel_endurance_hrs, destination_airport, origin_airport, clearance_type (hold/cleared_approach/cleared_landing/go_around/taxi/cleared_takeoff), runway_assigned, approach_type (ILS/VOR/RNAV/visual), separation_from_nearest_nm, wake_turbulence_category (light/medium/heavy/super), flight_level_status (climbing/descending/level), eta_minutes, delay_minutes, special_status (none/emergency/medical/minimum_fuel/VIP), remarks
- **Narrative voice**: ATC transcript / sector log ("14:32 — United 1523 descending through FL280 for FL180, speed 310 knots. Assigned heading 270 for radar vectors ILS 28R. Traffic 2 o'clock, 5 miles, same altitude, Delta 445 also descending.")
- **Scale**: 10-50 entities, 5-30 updates per aircraft per session
- **Constraints**: Vertical separation minimums (1000ft). Horizontal separation (3-5nm). Speed restrictions by altitude/airspace. Aircraft can't climb above service ceiling. Fuel is finite and decreasing.
- **Status**: Not started

### 19. Server / Network Monitoring
- **Entities**: Services or nodes (5-30)
- **Sample attributes**: status (healthy/degraded/critical/down/maintenance), cpu_pct, memory_pct, disk_pct, p50_latency_ms, p95_latency_ms, p99_latency_ms, error_rate_pct, request_rate_rps, active_connections, response_time_avg_ms, uptime_pct, last_deploy_timestamp, deploy_version, thread_count, gc_pause_ms, cache_hit_rate_pct, queue_depth, db_connection_pool_used, db_connection_pool_max, certificate_expiry_days, log_error_count_last_hour, alert_level (none/warning/critical/paging), replica_count, healthy_replica_count, container_restarts_24h, network_in_mbps, network_out_mbps
- **Narrative voice**: Incident report / status page ("14:32 UTC — auth-service-prod-03 latency spiked to 2,400ms p99, up from 85ms baseline. CPU at 94%, memory 78%. Error rate climbed to 3.2%. Scaled to 6 replicas at 14:35. By 14:41, latency recovered to 120ms.")
- **Scale**: 5-30 entities, 10-100 updates per incident window
- **Constraints**: CPU/memory 0-100%. Error rate increases when service degrades. Scaling takes time to take effect. Dependencies between services (upstream failure causes downstream degradation).
- **Status**: Not started

### 20. Weather Station Network
- **Entities**: Stations (5-20)
- **Sample attributes**: temperature_c, feels_like_c, dewpoint_c, barometric_pressure_hpa, pressure_trend (rising/steady/falling), wind_speed_kmh, wind_gust_kmh, wind_direction (N/NE/E/SE/S/SW/W/NW), precipitation_rate_mm_hr, precipitation_type (none/rain/snow/sleet/freezing_rain/hail), accumulated_precip_mm, humidity_pct, visibility_km, cloud_cover_pct, cloud_base_ft, uv_index, solar_radiation_wm2, soil_temperature_c, soil_moisture_pct, lightning_strikes_10min, station_status (online/degraded/offline/calibrating), battery_voltage
- **Narrative voice**: Weather summary / observer notes ("Station WX-Ridgecrest-04, 0600 local: Temperature dropped to -3.2°C overnight, pressure falling at 1008 hPa. Wind shifted from SW to NW at 28 km/h. Light snow at 0.8 mm/hr, visibility reduced to 4.2 km.")
- **Scale**: 5-20 entities, 24-96 readings per day (hourly or sub-hourly)
- **Constraints**: Temperature, pressure, humidity follow weather physics. Wind direction changes gradually (mostly). Precipitation correlates with pressure/humidity. Stations can go offline.
- **Status**: Not started

### 21. Supply Chain / Cold Chain
- **Entities**: Shipments (5-25)
- **Sample attributes**: current_location (city/facility), eta_datetime, original_eta, status (in_transit/customs_hold/at_warehouse/out_for_delivery/delivered/exception), container_temp_c, temp_excursion (boolean), temp_excursion_duration_min, humidity_pct, transit_mode (truck/rail/ocean/air), carrier_name, customs_cleared (boolean), seal_intact (boolean), days_in_transit, total_weight_kg, package_count, damage_reported (none/minor/major), delay_reason (none/weather/customs/mechanical/congestion/labor), tracking_updates_count, last_scan_location, distance_remaining_km, insurance_claimed (boolean)
- **Narrative voice**: Logistics updates ("Feb 14, 08:00 EST — Shipment SHP-4821 arrived Memphis hub. Container temp -18.3°C, within spec. Customs cleared at 06:15. ETA Nashville revised to Feb 15 due to weather delay.")
- **Scale**: 5-25 entities, 5-20 updates per shipment
- **Constraints**: Temperature excursions trigger alerts. Status transitions are ordered (can't go from delivered back to in_transit). ETA can only be revised, not reversed. Customs clearance happens once.
- **Status**: Not started

### 22. Autonomous Vehicle Fleet
- **Entities**: Vehicles (5-25)
- **Sample attributes**: battery_pct, location (named stop/street), speed_kmh, passenger_count, ride_status (idle/en_route/charging/maintenance/emergency_stop), next_stop, estimated_range_km, sensor_health (all_clear/lidar_degraded/camera_obstructed/gps_drift), ambient_conditions (clear/rain/fog/snow), override_events_today, total_rides_today, total_distance_today_km, charging_rate_kw, time_to_full_charge_min, cabin_temperature_c, doors_status (locked/unlocked), connectivity (online/degraded/offline), software_version, miles_since_last_maintenance, tire_pressure_psi, brake_pad_pct
- **Narrative voice**: Fleet dispatch log ("09:14 — AV-019 completed pickup at Elm Street Hub, 4 passengers, en route to Central Station. Battery 67%, range 142 km. AV-007 diverted to charging — battery at 14%.")
- **Scale**: 5-25 entities, 20-100 updates per day
- **Constraints**: Battery only decreases while driving, increases while charging. Passengers can only board when stopped. Speed limited by road/weather. Maintenance takes vehicle offline.
- **Status**: Not started

### 23. Greenhouse / Vertical Farm
- **Entities**: Crop beds or zones (5-25)
- **Sample attributes**: soil_moisture_pct, plant_height_cm, growth_stage (seedling/vegetative/flowering/fruiting/harvest_ready), pest_status (clear/aphids_low/aphids_high/whitefly/spider_mites/fungal_early/fungal_advanced), nutrient_ec (electrical conductivity), soil_ph, light_hours_per_day, temperature_c, humidity_pct, co2_ppm, yield_estimate_kg, days_since_planting, last_watered, last_fertilized, treatment_applied (none/neem_oil/pyrethrin/fungicide/fertilizer), leaf_color (green/yellowing/browning/spotted), root_health (healthy/root_bound/root_rot), pollination_status (not_needed/pending/complete), brix_reading (sugar content), water_usage_liters
- **Narrative voice**: Grower's daily log ("Feb 12: Bed 7A tomatoes at 42cm, entering early flowering. Soil moisture 55%. Spotted early aphids on Bed 3C — applied neem oil. Bed 12B lettuce harvest-ready at 28cm.")
- **Scale**: 5-25 entities, 10-30 updates per week
- **Constraints**: Growth stages are sequential. Height only increases (until harvest). Pest status worsens without treatment. Moisture depletes over time. Temperature affects growth rate.
- **Status**: Not started

### 24. Aquarium / Zoo
- **Entities**: Animals (5-30)
- **Sample attributes**: weight_kg, food_consumed_pct_of_offered, food_type, food_amount_grams, enclosure_name, health_status (healthy/observation/medicated/quarantine/recovering), medication (none/antibiotics/anti_parasitic/supplements/pain_relief), behavior (active/lethargic/aggressive/social/nesting/hiding), enrichment_response (high/moderate/disinterested), last_vet_check_date, companion_animals, fecal_quality (normal/loose/bloody/absent), body_condition_score (1-5), coat_or_skin_condition, eye_clarity, appetite (normal/increased/decreased/refusing), water_quality_ph (aquatic), water_temp_c (aquatic), breeding_status (non_breeding/courtship/mating/pregnant/nursing)
- **Narrative voice**: Keeper daily log ("Feb 15 — Koda (sea otter, M, age 6) ate 85% of morning feed. Weight steady at 28.4 kg. Moved to main Coast exhibit with Luna. Skin lesion on left forepaw resolved after antibiotic course.")
- **Scale**: 5-30 entities, 5-20 updates per week
- **Constraints**: Weight changes are gradual. Medication courses have durations. Enclosure transfers are discrete events. Breeding is seasonal for most species. Health deterioration leads to quarantine.
- **Status**: Not started

---

## OTHER DOMAINS (11 domains)

### 25. F1 / Motorsport Race
- **Entities**: Cars/drivers (20)
- **Sample attributes**: position, gap_to_leader_sec, interval_to_car_ahead_sec, lap_number, sector_1_time, sector_2_time, sector_3_time, lap_time, best_lap, tire_compound (soft/medium/hard/intermediate/wet), tire_age_laps, pit_stops_completed, pit_stop_duration_sec, drs_available (boolean), ers_deployment_pct, fuel_load_kg, speed_trap_kmh, penalties_sec, penalty_type (none/5sec/10sec/drive_through/stop_go), status (racing/in_pit/retired/dnf/dsq), damage (none/front_wing/rear_wing/suspension/floor), radio_messages, overtakes_completed, defending_position, fastest_lap_holder (boolean), points_projected
- **Narrative voice**: Race commentary ("Lap 34: Verstappen pitted from P1 for mediums, rejoining in P4 behind Hamilton. Gap to Leclerc in the lead now 12.3 seconds. Tire delta suggests the undercut should work by lap 40.")
- **Scale**: 20 entities, 50-70 laps, multiple updates per lap
- **Constraints**: Positions 1-20, no two cars same position. Tire compounds limited per race (must use 2 types). Pit stop takes ~20-25 seconds. DRS only in designated zones and within 1 second of car ahead. Fuel decreases linearly.
- **Status**: Not started

### 26. Fitness Training Camp
- **Entities**: Athletes (4-20)
- **Sample attributes**: resting_hr_bpm, session_avg_hr, max_hr_session, distance_km, pace_min_per_km, elevation_gain_m, cadence_spm, power_watts (cycling), training_load_tss, sleep_hours, sleep_quality (poor/fair/good/excellent), muscle_soreness (1-10), injury_status (clear/niggle/restricted/sidelined), body_weight_kg, body_fat_pct, hydration_status (well/mild_dehydration/dehydrated), vo2max_estimated, lactate_threshold, blood_oxygen_pct, mood_rating (1-10), rpe (rate of perceived exertion 1-10), recovery_score, hrv_ms
- **Narrative voice**: Coach's log ("Tuesday — Priya: 10km tempo at 4:32/km, avg HR 168. Hamstring tightness after km 7, backed off to 5:10. Soreness 6/10. Sleep 5.5hrs. Resting HR elevated at 58 (baseline 52). Recovery day recommended.")
- **Scale**: 4-20 entities, 5-30 sessions over training block
- **Constraints**: Fatigue accumulates. Poor sleep → elevated resting HR. Injury status limits training. Overtraining leads to performance decline. Progressive overload principle.
- **Status**: Not started

### 27. Archaeological Dig
- **Entities**: Trenches/excavation units (4-15)
- **Sample attributes**: current_depth_cm, target_depth_cm, stratum (topsoil/fill/occupation_layer_1/occupation_layer_2/sterile/bedrock), soil_type (sandy_loam/clay/gravel/ash/compact_fill/organic), soil_color (Munsell code or description), moisture (dry/moist/waterlogged), artifact_count_today, artifact_types (pottery/lithics/bone/metal/charcoal/glass/shell/none), diagnostic_artifacts_found, feature_identified (none/posthole/hearth/wall_foundation/pit/burial/floor_surface/ditch), feature_dimensions_cm, dating_estimate (era or range), c14_sample_taken (boolean), photo_documented (boolean), plan_drawn (boolean), finds_bag_count, soil_sample_taken (boolean), status (active/paused/backfilled/awaiting_specialist), excavator_name, weather_impact (none/rain_delay/frozen_ground/too_hot)
- **Narrative voice**: Field diary ("Day 14, Trench J-14: Excavated to 85cm depth, entering dense ash deposit. 12 pottery sherds recovered, coarseware. Possible hearth in southeast quadrant — paused for photo documentation.")
- **Scale**: 4-15 entities, 10-30 updates per excavation season
- **Constraints**: Depth only increases. Strata are sequential (you can't go from bedrock back to topsoil). Artifacts match their stratum period. Features are discovered, not created. Backfilled trenches stop updating.
- **Status**: Not started

### 28. Space Station Operations
- **Entities**: Subsystems (8-15) + crew members (3-7)
- **Sample attributes**: Per subsystem: power_generation_kw, battery_charge_pct, solar_array_angle_deg, thermal_loop_temp_c, coolant_flow_rate, cabin_pressure_psi, o2_level_pct, co2_level_mmhg, humidity_pct, ammonia_level_ppm, smoke_detector_status, water_recovery_rate_pct, attitude_control_status (nominal/degraded/manual), communication_link (active/scheduled_blackout/loss_of_signal), experiment_rack_status. Per crew: location_module, activity (sleep/exercise/maintenance/science/EVA/meal/communication), heart_rate, exercise_duration_today_min, radiation_dose_msv_cumulative, sleep_quality, calories_consumed
- **Narrative voice**: Mission control log ("ISS Flight Day 47: Solar array 2A angle adjusted to 52 degrees for optimal power generation, battery charge at 88%. CO2 scrubber in Node 3 showing reduced efficiency — crew performed filter replacement. Commander Tanaka completed 2-hour exercise session on ARED.")
- **Scale**: 10-20 entities, 20-50 updates per day
- **Constraints**: Power must balance generation and consumption. Attitude adjustments affect solar array output. EVAs require depressurization. Communication has scheduled blackout windows. Consumables deplete.
- **Status**: Not started

### 29. Cooking Competition
- **Entities**: Contestants (4-20)
- **Sample attributes**: status (competing/eliminated/winner), current_round, dish_name, dish_description, time_remaining_sec, basket_ingredients (list), basket_ingredients_used (list of booleans), cook_stage (prep/cooking/plating/presented), protein_internal_temp_f, seasoning_level (under/balanced/over), presentation_score (1-10), taste_score (1-10), creativity_score (1-10), total_score, judge_comments, rounds_survived, advantages_held, penalties, station_number, equipment_malfunction (boolean), injury (none/cut/burn)
- **Narrative voice**: Competition recap ("With 8 minutes remaining, Chef Tanaka's risotto was still soupy while her competitors were already plating. She cranked the heat and stirred frantically. Judge Morrison noted her sea bass was perfectly seared at 138°F internal, but the risotto was going to make or break her round.")
- **Scale**: 4-20 entities, 5-20 updates per round, 2-4 rounds
- **Constraints**: Time only decreases. Ingredients must come from basket. Eliminated contestants don't return (usually). Scores are per-round. Cook stages are sequential.
- **Status**: Not started

### 30. Construction Site
- **Entities**: Tasks/buildings/zones (5-50)
- **Sample attributes**: status (not_started/in_progress/blocked/completed/on_hold/rework), completion_pct, planned_start, planned_end, actual_start, actual_end, days_ahead_or_behind, assigned_crew, worker_count, phase (demolition/excavation/foundation/framing/roofing/electrical/plumbing/HVAC/drywall/finishing/inspection/landscaping), budget_allocated, budget_spent, budget_variance_pct, materials_on_site (boolean), materials_ordered_eta, inspection_status (not_needed/scheduled/passed/failed/rework_needed), safety_incidents, weather_delay_days, change_orders, rfi_count (requests for information), punch_list_items, subcontractor_name
- **Narrative voice**: Project manager daily log ("Phase 2 framing on Building C hit 65% completion, up from 48% yesterday. Crew of 12 making good progress despite morning rain delay. Electrical rough-in on Building A passed inspection — moving to drywall next week. Building B foundation rework still blocked waiting on engineering approval.")
- **Scale**: 5-50 entities, daily updates over weeks/months
- **Constraints**: Phases are sequential. Can't start framing before foundation. Inspections gate next phase. Weather affects outdoor work. Budget overruns are cumulative.
- **Status**: Not started

### 31. Stock Trading Desk
- **Entities**: Positions/traders (5-30)
- **Sample attributes**: Per position: ticker, shares_held, avg_cost_basis, current_price, unrealized_pnl, realized_pnl, day_high, day_low, day_open, pct_change, volume, position_size_pct_of_portfolio. Per trader: cash_balance, total_portfolio_value, buying_power, margin_used, win_rate_pct, trades_today, biggest_winner, biggest_loser, sharpe_ratio, max_drawdown_pct, risk_limit_used_pct, open_orders_count
- **Narrative voice**: Trading floor log / desk summary ("10:32 — Desk 4 (Chen) added 500 shares NVDA at $892, bringing position to 1,200 shares at $876 avg. Unrealized P&L now +$19,200. Portfolio up 1.3% on the day. Risk utilization at 74% of limit.")
- **Scale**: 5-30 entities, 20-200 trades per day
- **Constraints**: Can't sell more shares than held (unless short). Cash balance decreases on buys, increases on sells. P&L follows price changes. Margin limits constrain position sizes.
- **Status**: Not started

### 32. Fire Incident Command
- **Entities**: Incidents/units (3-15)
- **Sample attributes**: incident_type (structure_fire/wildfire/vehicle_accident/hazmat/medical/rescue/gas_leak/flooding), alarm_level (1-5), units_assigned, personnel_on_scene, area_affected (sq ft or acres), casualties (injured count), fatalities, evacuation_radius, evacuation_count, water_supply_status (hydrant/tanker/draft), fire_behavior (smoke_showing/working_fire/fully_involved/under_control/overhaul/extinguished), hazmat_identified, road_closures, mutual_aid_requested (boolean), incident_commander, elapsed_time_minutes, weather_conditions, wind_speed, wind_direction, exposures_threatened, damage_estimate_usd, cause_determined (boolean)
- **Narrative voice**: Dispatch / incident log (already proven — "At 8:15 PM, Engine Company 7 responded to a structure fire at 742 Elm Street. The fire captain radioed that flames had jumped to 744 Elm.")
- **Scale**: 3-15 entities, 5-30 updates per incident over hours
- **Constraints**: Alarm levels only increase. Casualties can only increase. Fire behavior follows progression (smoke→working→fully_involved→under_control→extinguished). Resources are finite.
- **Status**: Proven — 10 stories in existing dataset

### 33. Emergency Room Triage
- **Entities**: Patients (5-30)
- **Sample attributes**: chief_complaint, triage_level (1-5, ESI scale), arrival_time, time_in_waiting_min, time_in_treatment_min, assigned_bed, assigned_provider, vitals_hr, vitals_bp_systolic, vitals_bp_diastolic, vitals_rr, vitals_spo2, vitals_temp, pain_level (0-10), gcs_score, acuity_reassessment, labs_ordered (boolean), labs_resulted (boolean), imaging_ordered (none/xray/ct/mri/ultrasound), imaging_resulted (boolean), iv_access (boolean), medications_given, consult_requested (none/surgery/cardiology/neurology/ortho/psych), disposition (pending/admit/discharge/transfer/ama/expired), diagnosis, time_to_provider_min, time_to_disposition_min
- **Narrative voice**: ER board / charge nurse notes ("Bed 4: 45-year-old male, chest pain, ESI level 2. Arrived 14:20, seen by Dr. Patel at 14:28. Troponin pending. 12-lead EKG shows ST elevation in leads V1-V4. Cardiology activated, cath lab on standby. BP 168/94, HR 102, SpO2 97%.")
- **Scale**: 5-30 entities, 3-15 updates per patient per visit
- **Constraints**: Triage levels rarely change (but can if condition worsens). Labs take time to result. Disposition is usually final. Time-to-provider is a key metric. STEMI/stroke have time-critical protocols.
- **Status**: Not started

### 34. Music Festival Production
- **Entities**: Stages (3-8) + acts (15-40) + equipment systems (5-15)
- **Sample attributes**: Per stage: current_act, next_act, set_start_time, set_end_time, changeover_status (in_progress/complete), sound_level_db, crowd_size_estimate, crowd_density, weather_exposure. Per act: status (scheduled/soundcheck/performing/completed/cancelled), set_duration_min, songs_played, songs_remaining, technical_issues (none/monitor_feedback/mic_cut/instrument_failure/power_dip), crowd_response (low/moderate/high/electric), merch_sales_during_set. Per equipment: power_load_kw, generator_fuel_pct, pa_system_status (online/degraded/offline), lighting_rig_status, video_wall_status, backup_generator_status
- **Narrative voice**: Production manager log ("20:45 — Main Stage: Headliner 'Midnight Echo' opened with 'Stardust Highway' to an estimated 35,000. Sound level at 102dB, within permit. Monitor engineer reports feedback on channel 7 — resolved by 20:48. Side Stage changeover running 4 minutes behind schedule.")
- **Scale**: 10-40 entities, 20-100 updates over a festival day
- **Constraints**: Acts can't overlap on same stage. Changeovers take minimum time. Power has maximum load. Generator fuel depletes. Weather affects outdoor stages.
- **Status**: Not started

### 35. Film Production Tracking
- **Entities**: Scenes/setups + crew departments + equipment
- **Sample attributes**: Per scene: scene_number, status (not_started/rehearsal/lighting/rolling/printed/moving_on), takes_shot, takes_printed, estimated_duration_min, actual_duration_min, location, call_time, first_shot_time, wrap_time, actors_in_scene, extras_count. Per department: department_name (camera/sound/lighting/art/wardrobe/makeup/stunts), crew_count, overtime_hours, equipment_status, next_setup_ready. Global: pages_shot_today, pages_remaining, schedule_day, total_shoot_days, budget_spent_today, meals_served, weather_status, union_break_compliance
- **Narrative voice**: Assistant director daily report ("Day 14 of 32: Completed scenes 47, 48, and the first half of scene 52. 4.2 pages shot (target was 3.8). First shot at 07:42, lunch at 13:00, wrap at 19:15. Rain delay cost 45 minutes on the exterior setup. Stunts department needs additional rigging time for tomorrow's car chase — requesting 30-minute early call for unit.")
- **Scale**: 10-30 entities, 20-50 updates per shooting day
- **Constraints**: Scenes require specific actors (availability). Weather affects exteriors. Union rules mandate breaks. Budget is finite. Equipment failures delay production.
- **Status**: Not started

---

## DOMAINS STILL TO CONSIDER

If we need more domains, potential additions:
- **Beehive monitoring** (3-12 hives: weight, temp, queen status, varroa count, activity)
- **Submarine operations** (systems: ballast, depth, reactor, sonar, torpedoes, crew stations)
- **Prison/correctional facility** (inmates: block assignment, behavior incidents, privileges, work detail)
- **Airline operations center** (flights: delay status, gate, crew, fuel, passengers, connections)
- **Power grid operations** (generators, transmission lines, load zones: frequency, voltage, demand)
- **Political campaign** (candidates: poll numbers, fundraising, endorsements, ad spend, favorability)
- **Disaster relief coordination** (shelters, supply depots, teams: capacity, supplies, personnel)
- **Ocean research vessel** (instruments, stations, samples: depth, temperature, salinity, specimens)
- **Competitive esports team season** (players: win rate, role, champion pool, scrim performance)
- **Smart city traffic management** (intersections: flow rate, congestion, signal timing, incidents)

---

## NEXT STEPS

For each approved domain, we will create a detailed spec covering:
1. **Complete attribute list** with exact value ranges and realistic distributions
2. **Value pools** for categorical attributes (item names, location names, etc.)
3. **Domain constraints** (state machine rules for valid transitions)
4. **Narrative templates** (sentence patterns for each type of state change)
5. **Entity name pools** (character names, callsigns, batch IDs, etc.)
6. **Example narrative** showing what a generated story looks like

We go through these one domain at a time.
