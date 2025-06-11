from flask import Flask, render_template, request, url_for, redirect, abort, flash, session, make_response
from pymongo import MongoClient
from bson import ObjectId, Int64
from datetime import datetime
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_1234")

client = MongoClient("mongodb://localhost:27017/")
db = client.Research

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        user = db.user.find_one({'email': email})
        if not user:
            flash('No such user. Please register first.', 'error')
            return redirect(url_for('login'))
        
        if not check_password_hash(user['password'], password):
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
        
        session['user_id'] = str(user['_id'])
        flash(f'Welcome back, {email}!', 'success')
        return redirect(url_for('projects'))
    
    return render_template('login.html')  

@app.route('/logout')
def logout():
    print("Logout called")
    session.clear()
    print("Session cleared")
    flash('You have been logged out', 'info')
    print("About to redirect to login")
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    print("Registering user")
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        email = request.form['email']
        print(f"Registering user: {name}, email: {email}")
        if db.user.find_one({'name':  name}):
            flash('Name already exists', 'error')
            return render_template('register.html')
        else:
            hashed_pw = generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)
            db.user.insert_one({'name': name, 'email': email, 'password': hashed_pw, "created_at": datetime.now()})
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for("login"))
        
@app.route('/', methods=['GET'])
def index():
    if session.get('user_id'):
        return redirect(url_for('projects'))
    return redirect(url_for('login'))

@app.route('/projects', methods=(['GET']))
def projects():
    all_projects = db.projects
    get_all_projects = all_projects.find() 
    return render_template('projects.html', projects=get_all_projects)

def fetch_tweets_with_topics(db, logger=None):
    """
    Fetch tweets and their topics from topic collection.
    Matches NumberLong IDs between topic collection and tweet collection.
    Returns all tweet attributes for feed reconstruction.
    
    Args:
        db: MongoDB database connection
        logger: optional logger for debug messages
    
    Returns:
        List of dicts with all tweet attributes plus topic information.
    """
    tweets = []
    
    # Find all topic documents
    topic_docs = db.topic.find({})
    
    for doc in topic_docs:
        # Get the document array
        documents = doc.get("document", [])
        if not documents:
            if logger:
                logger.debug(f"No 'document' field found or empty in document id={doc.get('_id')}")
            continue
        
        for item in documents:
            # Extract id from the item
            id_field = item.get("id")
            if id_field is None:
                if logger:
                    logger.debug(f"Skipping item with missing 'id': {item}")
                continue
            
            # Convert id to string format for querying id_str
            tweet_id_str = None
            
            if isinstance(id_field, dict) and "$numberLong" in id_field:
                # Handle MongoDB NumberLong format: {"$numberLong": "1903424019937792172"}
                tweet_id_str = id_field["$numberLong"]
            else:
                # Handle other formats (convert to string)
                tweet_id_str = str(id_field)
            
            if not tweet_id_str:
                if logger:
                    logger.debug(f"Could not extract id string from: {id_field}")
                continue
            
            # Query the tweet collection using id_str field
            # Convert string to int for proper NumberLong matching
            try:
                tweet_id_int = int(tweet_id_str)
                tweet_doc = db.tweet.find_one({'id_str': tweet_id_int})
            except ValueError:
                if logger:
                    logger.debug(f"Could not convert id_str to int: {tweet_id_str}")
                continue
            
            if tweet_doc:
                # Create complete tweet object with all attributes
                tweet_data = {
                    # Core tweet identifiers
                    '_id': tweet_doc.get('_id'),
                    'id_str': tweet_doc.get('id_str'),
                    'conversation_id_str': tweet_doc.get('conversation_id_str'),
                    
                    # Tweet content
                    'full_text': tweet_doc.get('full_text') or tweet_doc.get('text', ''),
                    'lang': tweet_doc.get('lang'),
                    
                    # User information
                    'user_id_str': tweet_doc.get('user_id_str'),
                    'username': tweet_doc.get('username'),
                    'location': tweet_doc.get('location'),
                    
                    # Timestamp
                    'created_at': tweet_doc.get('created_at'),
                    
                    # Engagement metrics
                    'favorite_count': tweet_doc.get('favorite_count', 0),
                    'retweet_count': tweet_doc.get('retweet_count', 0),
                    'reply_count': tweet_doc.get('reply_count', 0),
                    'quote_count': tweet_doc.get('quote_count', 0),
                    
                    # URLs and links
                    'tweet_url': tweet_doc.get('tweet_url'),
                    
                    # Reply information
                    'in_reply_to_screen_name': tweet_doc.get('in_reply_to_screen_name'),
                    'in_reply_to_status_id_str': tweet_doc.get('in_reply_to_status_id_str'),
                    'in_reply_to_user_id_str': tweet_doc.get('in_reply_to_user_id_str'),
                    
                    # Media and entities (if present)
                    'media': tweet_doc.get('media'),
                    'entities': tweet_doc.get('entities'),
                    'extended_entities': tweet_doc.get('extended_entities'),
                    
                    # Retweet information (if present)
                    'retweeted_status': tweet_doc.get('retweeted_status'),
                    'is_quote_status': tweet_doc.get('is_quote_status'),
                    'quoted_status': tweet_doc.get('quoted_status'),
                    'quoted_status_id_str': tweet_doc.get('quoted_status_id_str'),
                    
                    # Additional metadata
                    'source': tweet_doc.get('source'),
                    'possibly_sensitive': tweet_doc.get('possibly_sensitive'),
                    'withheld_scope': tweet_doc.get('withheld_scope'),
                    'withheld_copyright': tweet_doc.get('withheld_copyright'),
                    'withheld_in_countries': tweet_doc.get('withheld_in_countries'),
                    
                    # Topic information from topic collection
                    'topic': item.get('topic'),
                    'topic_score': item.get('score'),  # if available
                    'topic_keywords': item.get('keywords'),  # if available
                    
                    # Any other fields that might be present
                    **{k: v for k, v in tweet_doc.items() if k not in [
                        '_id', 'id_str', 'conversation_id_str', 'full_text', 'text', 'lang',
                        'user_id_str', 'username', 'location', 'created_at', 'favorite_count',
                        'retweet_count', 'reply_count', 'quote_count', 'tweet_url',
                        'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str',
                        'media', 'entities', 'extended_entities', 'retweeted_status', 'is_quote_status',
                        'quoted_status', 'quoted_status_id_str', 'source', 'possibly_sensitive',
                        'withheld_scope', 'withheld_copyright', 'withheld_in_countries'
                    ]}
                }
                
                tweets.append(tweet_data)
            else:
                if logger:
                    logger.debug(f"No tweet found with id_str={tweet_id_int}")
    
    print(tweets)
    return tweets

@app.route('/dashboard/<project_id>', methods=(['GET']))
def dashboard(project_id):
    try:
        pid = ObjectId(project_id)
    except:
        abort(404)
        
    topics_doc = db.topic.find_one({'_id': pid})

    if topics_doc is None:
        print(f"No topic document found for project_id={pid}")
    else:
        topics_data = topics_doc.get("word", {}).get("array", [])
        # Proceed if topics_data is valid
        if not topics_data:
            print("No topics data found in the document.")
        else:
            topics = sorted(topics_data, key=lambda x: x["interpretation"])
            for topic in topics:
                topic_id = topic["interpretation"]
                keywords = topic["keywords"]
                # print(f"Topic {topic_id}: {keywords}")
    
    # print(topics)
    
    print(topics)
    # 2) fetch tweets + their dominant topic
    #    here we join tweets_topics → tweets_raw
    tweets = fetch_tweets_with_topics(db, logger=app.logger)
    print(tweets)

    
    return render_template(
        'dashboard.html',
        project_id=project_id,
        topics=topics,
        tweets=tweets
    )

@app.route('/create_project')
def create_project():
    return render_template('create_project.html')

