import sqlite3


def create_table():
    conn = sqlite3.connect('burnout_history.db')
    c = conn.cursor()
    # 11 Columns total: ID + 8 Features + Risk + Timestamp
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                ( id INTEGER PRIMARY KEY AUTOINCREMENT,
                  day_type TEXT,
                  work_hours REAL,
                  screen_time REAL,
                  meetings_count INTEGER,
                  breaks_taken INTEGER,
                  after_hours_work INTEGER,
                  sleep_hours REAL,
                  task_completion REAL,
                  burnout_risk TEXT,
                  timestamp DATETIME DEFAULT ( datetime ( 'now', 'localtime' )))''')
    conn.commit()
    conn.close()


def add_prediction(day_type, wh, st, mc, bt, ahw, sh, tc, risk):
    conn = sqlite3.connect('burnout_history.db')
    c = conn.cursor()
    # Inserting values in the exact order they appear in the UI
    c.execute('''INSERT INTO history (day_type, work_hours, screen_time,
                                      meetings_count, breaks_taken, after_hours_work, sleep_hours,
                                      task_completion, burnout_risk)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (day_type, wh, st, mc, bt, ahw, sh, tc, risk))
    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect('burnout_history.db')
    c = conn.cursor()
    # Order by newest first so the professor sees your latest tests at the top
    c.execute('SELECT * FROM history ORDER BY id DESC')
    data = c.fetchall()
    conn.close()
    return data