from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Read the data from the CSV file
df = pd.read_csv('files/result_new.csv')

@app.route('/')
def index():
    # Get unique hospital names for dropdown
    hospital_names = df['Org Name'].unique()

    # Get the filtering criteria from the query parameters
    filter_criteria = request.args.get('filter_criteria')

    # Apply filtering if criteria is provided
    if filter_criteria:
        filtered_df = df[df['Org Name'] == filter_criteria]
        comments = filtered_df.to_dict(orient='records')
    else:
        comments = df.to_dict(orient='records')

    # Pagination
    page = int(request.args.get('page', 1))
    start_index = (page - 1) * 25
    end_index = start_index + 25
    comments_on_page = comments[start_index:end_index]

    # Calculate the total number of pages
    total_pages = (len(comments) + 24) // 25  # Adding 24 to round up to the nearest integer

    return render_template('index_cloud.html', comments=comments_on_page, page=page, total_pages=total_pages, hospital_names=hospital_names)


if __name__ == '__main__':
    app.run(debug=True)
