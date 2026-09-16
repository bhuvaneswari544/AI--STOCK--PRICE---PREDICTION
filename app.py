import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
from io import StringIO
from flask import Flask, render_template, request, Response, jsonify
from data_loader import get_stock_data, search_global_companies
from model import predict_stock

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    summary = None
    error = None
    prediction_info = None
    chart_data_json = "{}"
    forecast_chart_json = "{}"
    history_records = None
    selected_period = '1y'

    if request.method == 'POST':
        query = request.form.get('stock', '').strip()
        selected_period = request.form.get('period', '1y')

        if not query:
            error = "Please enter a company name or ticker symbol."
        else:
            try:
                # 1. Fetch live stock data in ₹ with selected timeframe
                data_result = get_stock_data(query, period=selected_period)
                df = data_result['df']
                summary = data_result['summary']
                chart_data = data_result['chart_data']
                history_records = data_result['history_records']

                # 2. Run 7-day AI LSTM Prediction
                prediction_info = predict_stock(summary['ticker'], df, days_ahead=7)

                chart_data_json = json.dumps(chart_data)
                
                # Combine last historical date with 7-day forecast for connected dotted line
                last_hist_date = chart_data['dates'][-1]
                last_hist_close = chart_data['close'][-1]
                
                forecast_dates = [last_hist_date] + [item['iso_date'] for item in prediction_info['forecast_items']]
                forecast_prices = [last_hist_close] + [item['price'] for item in prediction_info['forecast_items']]

                forecast_chart_json = json.dumps({
                    'dates': forecast_dates,
                    'prices': forecast_prices
                })

            except Exception as e:
                print(f"Error analyzing {query}: {e}")
                error = f"Could not find stock data for '{query}'. Please verify the company name and try again."

    return render_template(
        'index.html',
        summary=summary,
        error=error,
        prediction_info=prediction_info,
        chart_data_json=chart_data_json,
        forecast_chart_json=forecast_chart_json,
        history_records=history_records,
        selected_period=selected_period
    )

@app.route('/download-csv/<ticker>')
def download_csv(ticker):
    """Generates and downloads historical stock data in ₹ as a CSV file."""
    try:
        data_result = get_stock_data(ticker, period='1y')
        df = data_result['df']
        
        output = StringIO()
        output.write("Date,Open (INR),High (INR),Low (INR),Close (INR),Volume\n")
        for dt, row in df.iloc[::-1].iterrows():
            vol = int(row['Volume']) if not row.isna()['Volume'] else 0
            output.write(f"{dt.strftime('%Y-%m-%d')},{row['Open']},{row['High']},{row['Low']},{row['Close']},{vol}\n")
        
        csv_content = output.getvalue()
        output.close()
        
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={ticker}_history_INR.csv"}
        )
    except Exception as e:
        return f"Error exporting CSV: {e}", 400

if __name__ == '__main__':
    print("\n🚀 AI Stock Dashboard running at: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)