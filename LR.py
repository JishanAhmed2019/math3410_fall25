import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import time

plt.style.use(['science', 'no-latex'])

# ----------------------------
# DATA (columns 2 = spending, 4 = sales)
# ----------------------------
data_string = """
1,230.1,37.8,69.2,22.1
2,44.5,39.3,45.1,10.4
3,17.2,45.9,69.3,9.3
4,151.5,41.3,58.5,18.5
5,180.8,10.8,58.4,12.9
6,8.7,48.9,75,7.2
7,57.5,32.8,23.5,11.8
8,120.2,19.6,11.6,13.2
9,8.6,2.1,1,4.8
10,199.8,2.6,21.2,10.6
11,66.1,5.8,24.2,8.6
12,214.7,24,4,17.4
13,23.8,35.1,65.9,9.2
14,97.5,7.6,7.2,9.7
15,204.1,32.9,46,19
16,195.4,47.7,52.9,22.4
17,67.8,36.6,114,12.5
18,281.4,39.6,55.8,24.4
19,69.2,20.5,18.3,11.3
20,147.3,23.9,19.1,14.6
21,218.4,27.7,53.4,18
22,237.4,5.1,23.5,12.5
23,13.2,15.9,49.6,5.6
24,228.3,16.9,26.2,15.5
25,62.3,12.6,18.3,9.7
26,262.9,3.5,19.5,12
27,142.9,29.3,12.6,15
28,240.1,16.7,22.9,15.9
29,248.8,27.1,22.9,18.9
30,70.6,16,40.8,10.5
31,292.9,28.3,43.2,21.4
32,112.9,17.4,38.6,11.9
33,97.2,1.5,30,9.6
34,265.6,20,0.3,17.4
35,95.7,1.4,7.4,9.5
36,290.7,4.1,8.5,12.8
37,266.9,43.8,5,25.4
38,74.7,49.4,45.7,14.7
39,43.1,26.7,35.1,10.1
40,228,37.7,32,21.5
41,202.5,22.3,31.6,16.6
42,177,33.4,38.7,17.1
43,293.6,27.7,1.8,20.7
44,206.9,8.4,26.4,12.9
45,25.1,25.7,43.3,8.5
46,175.1,22.5,31.5,14.9
47,89.7,9.9,35.7,10.6
48,239.9,41.5,18.5,23.2
49,227.2,15.8,49.9,14.8
50,66.9,11.7,36.8,9.7
51,199.8,3.1,34.6,11.4
52,100.4,9.6,3.6,10.7
53,216.4,41.7,39.6,22.6
54,182.6,46.2,58.7,21.2
55,262.7,28.8,15.9,20.2
56,198.9,49.4,60,23.7
57,7.3,28.1,41.4,5.5
58,136.2,19.2,16.6,13.2
59,210.8,49.6,37.7,23.8
60,210.7,29.5,9.3,18.4
61,53.5,2,21.4,8.1
62,261.3,42.7,54.7,24.2
63,239.3,15.5,27.3,15.7
64,102.7,29.6,8.4,14
65,131.1,42.8,28.9,18
66,69,9.3,0.9,9.3
67,31.5,24.6,2.2,9.5
68,139.3,14.5,10.2,13.4
69,237.4,27.5,11,18.9
70,216.8,43.9,27.2,22.3
71,199.1,30.6,38.7,18.3
72,109.8,14.3,31.7,12.4
73,26.8,33,19.3,8.8
74,129.4,5.7,31.3,11
75,213.4,24.6,13.1,17
76,16.9,43.7,89.4,8.7
77,27.5,1.6,20.7,6.9
78,120.5,28.5,14.2,14.2
79,5.4,29.9,9.4,5.3
80,116,7.7,23.1,11
81,76.4,26.7,22.3,11.8
82,239.8,4.1,36.9,12.3
83,75.3,20.3,32.5,11.3
84,68.4,44.5,35.6,13.6
85,213.5,43,33.8,21.7
86,193.2,18.4,65.7,15.2
87,76.3,27.5,16,12
88,110.7,40.6,63.2,16
89,88.3,25.5,73.4,12.9
90,109.8,47.8,51.4,16.7
91,134.3,4.9,9.3,11.2
92,28.6,1.5,33,7.3
93,217.7,33.5,59,19.4
94,250.9,36.5,72.3,22.2
95,107.4,14,10.9,11.5
96,163.3,31.6,52.9,16.9
97,197.6,3.5,5.9,11.7
98,184.9,21,22,15.5
99,289.7,42.3,51.2,25.4
100,135.2,41.7,45.9,17.2
101,222.4,4.3,49.8,11.7
102,296.4,36.3,100.9,23.8
103,280.2,10.1,21.4,14.8
104,187.9,17.2,17.9,14.7
105,238.2,34.3,5.3,20.7
106,137.9,46.4,59,19.2
107,25,11,29.7,7.2
108,90.4,0.3,23.2,8.7
109,13.1,0.4,25.6,5.3
110,255.4,26.9,5.5,19.8
111,225.8,8.2,56.5,13.4
112,241.7,38,23.2,21.8
113,175.7,15.4,2.4,14.1
114,209.6,20.6,10.7,15.9
115,78.2,46.8,34.5,14.6
116,75.1,35,52.7,12.6
117,139.2,14.3,25.6,12.2
118,76.4,0.8,14.8,9.4
119,125.7,36.9,79.2,15.9
120,19.4,16,22.3,6.6
121,141.3,26.8,46.2,15.5
122,18.8,21.7,50.4,7
123,224,2.4,15.6,11.6
124,123.1,34.6,12.4,15.2
125,229.5,32.3,74.2,19.7
126,87.2,11.8,25.9,10.6
127,7.8,38.9,50.6,6.6
128,80.2,0,9.2,8.8
129,220.3,49,3.2,24.7
130,59.6,12,43.1,9.7
131,0.7,39.6,8.7,1.6
132,265.2,2.9,43,12.7
133,8.4,27.2,2.1,5.7
134,219.8,33.5,45.1,19.6
135,36.9,38.6,65.6,10.8
136,48.3,47,8.5,11.6
137,25.6,39,9.3,9.5
138,273.7,28.9,59.7,20.8
139,43,25.9,20.5,9.6
140,184.9,43.9,1.7,20.7
141,73.4,17,12.9,10.9
142,193.7,35.4,75.6,19.2
143,220.5,33.2,37.9,20.1
144,104.6,5.7,34.4,10.4
145,96.2,14.8,38.9,11.4
146,140.3,1.9,9,10.3
147,240.1,7.3,8.7,13.2
148,243.2,49,44.3,25.4
149,38,40.3,11.9,10.9
150,44.7,25.8,20.6,10.1
151,280.7,13.9,37,16.1
152,121,8.4,48.7,11.6
153,197.6,23.3,14.2,16.6
154,171.3,39.7,37.7,19
155,187.8,21.1,9.5,15.6
156,4.1,11.6,5.7,3.2
157,93.9,43.5,50.5,15.3
158,149.8,1.3,24.3,10.1
159,11.7,36.9,45.2,7.3
160,131.7,18.4,34.6,12.9
161,172.5,18.1,30.7,14.4
162,85.7,35.8,49.3,13.3
163,188.4,18.1,25.6,14.9
164,163.5,36.8,7.4,18
165,117.2,14.7,5.4,11.9
166,234.5,3.4,84.8,11.9
167,17.9,37.6,21.6,8
168,206.8,5.2,19.4,12.2
169,215.4,23.6,57.6,17.1
170,284.3,10.6,6.4,15
171,50,11.6,18.4,8.4
172,164.5,20.9,47.4,14.5
173,19.6,20.1,17,7.6
174,168.4,7.1,12.8,11.7
175,222.4,3.4,13.1,11.5
176,276.9,48.9,41.8,27
177,248.4,30.2,20.3,20.2
178,170.2,7.8,35.2,11.7
179,276.7,2.3,23.7,11.8
180,165.6,10,17.6,12.6
181,156.6,2.6,8.3,10.5
182,218.5,5.4,27.4,12.2
183,56.2,5.7,29.7,8.7
184,287.6,43,71.8,26.2
185,253.8,21.3,30,17.6
186,205,45.1,19.6,22.6
187,139.5,2.1,26.6,10.3
188,191.1,28.7,18.2,17.3
189,286,13.9,3.7,15.9
190,18.7,12.1,23.4,6.7
191,39.5,41.1,5.8,10.8
192,75.5,10.8,6,9.9
193,17.2,4.1,31.6,5.9
194,166.8,42,3.6,19.6
195,149.7,35.6,6,17.3
196,38.2,3.7,13.8,7.6
197,94.2,4.9,8.1,9.7
198,177,9.3,6.4,12.8
199,283.6,42,66.2,25.5
200,232.1,8.6,8.7,13.4
"""

lines = data_string.strip().split("\n")
x_vals = []
y_vals = []
for line in lines:
    parts = line.split(",")
    x_vals.append(float(parts[2]))
    y_vals.append(float(parts[4]))
x = np.array(x_vals)
y = np.array(y_vals)

# ----------------------------
# LOSS FUNCTIONS
# ----------------------------
def mse_loss(x, y, w, b):
    return np.mean((y - (w * x + b)) ** 2)

def grad_w(x, y, w, b):
    return -2 * np.mean(x * (y - (w * x + b)))

def grad_b(x, y, w, b):
    return -2 * np.mean(y - (w * x + b))

# ----------------------------
# PRECOMPUTE LOSS SURFACE (3D)
# ----------------------------
@st.cache_data
def get_loss_surface():
    N = 25
    w_vals = np.linspace(0, 0.5, N)
    b_vals = np.linspace(0, 15, N)
    W, B = np.meshgrid(w_vals, b_vals)
    E = np.array([[mse_loss(x, y, wi, bi) for wi in w_vals] for bi in b_vals])
    return W, B, E

W, B, E = get_loss_surface()

# ----------------------------
# GRADIENT DESCENT FUNCTION
# ----------------------------
def run_gradient_descent(learning_rate, max_iter):
    """Run gradient descent and return complete history at each iteration"""
    w, b = 0.0, 0.0
    history = [(w, b, mse_loss(x, y, w, b))]
    
    for i in range(max_iter):
        dw = grad_w(x, y, w, b)
        db = grad_b(x, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        history.append((w, b, mse_loss(x, y, w, b)))
    
    return history

# ----------------------------
# PLOTTING FUNCTION
# ----------------------------
def create_plot(history, show_equation=True):
    """Create the visualization plot"""
    hist = np.array(history)
    w, b = hist[-1, 0], hist[-1, 1]
    current_iter = len(history) - 1
    
    fig = plt.figure(figsize=(14, 5))
    
    # Left: Data + regression line
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(x, y, color='#1f77b4', alpha=0.6, s=20, label='Data')
    x_plot = np.linspace(x.min(), x.max(), 100)
    y_plot = w * x_plot + b
    ax1.plot(x_plot, y_plot, color='red', linewidth=2, label=f'Fit Line')
    
    # Add equation annotation inside the plot
    if show_equation:
        equation_text = f'$y = {w:.4f}x + {b:.3f}$'
        ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel("Spending")
    ax1.set_ylabel("Sales")
    ax1.set_title(f"Linear Regression Fit (Iteration {current_iter})")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Right: 3D loss surface + GD path
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(W, B, E, cmap='viridis', alpha=0.7, edgecolor='none')
    
    # Plot GD trajectory
    ax2.plot(hist[:, 0], hist[:, 1], hist[:, 2], color='red', linewidth=2.5, marker='o', markersize=3, label='GD Path')
    
    # Highlight current position
    ax2.scatter([hist[-1, 0]], [hist[-1, 1]], [hist[-1, 2]], 
                color='yellow', s=100, edgecolor='black', linewidth=2, zorder=10)
    
    ax2.set_xlabel("Slope (w)")
    ax2.set_ylabel("Intercept (b)")
    ax2.set_zlabel("MSE Loss")
    ax2.set_title("Loss Surface & Gradient Descent Path")
    ax2.view_init(elev=20, azim=-40)
    
    return fig, hist[-1, 2]

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Gradient Descent: Linear Regression", layout="wide")
st.title("📉 Gradient Descent for Linear Regression")
st.markdown("""
This app demonstrates **gradient descent** to find the best-fit line for advertising data  
by minimizing the **Mean Squared Error (MSE)**. Choose from different visualization modes below.
""")

# Initialize session state
if 'full_history' not in st.session_state:
    st.session_state.full_history = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

col_controls, col_output = st.columns([1, 2])

with col_controls:
    st.subheader("⚙️ Parameters")
    
    # Remove key parameters that might cause conflicts
    learning_rate = st.slider(
        "Learning Rate (η)", 
        min_value=0.00001, 
        max_value=0.001, 
        value=0.00005, 
        step=0.00001,
        format="%.5f"
    )
    
    max_iter = st.slider(
        "Max Iterations", 
        min_value=100, 
        max_value=10000, 
        value=2000, 
        step=100
    )
    
    # Add step size for visualization
    step_increment = st.slider(
        "Steps per Click", 
        min_value=1, 
        max_value=200, 
        value=50, 
        step=10,
        help="Number of iterations to advance with each click"
    )
    
    # Auto-reset when parameters change
    if 'prev_lr' not in st.session_state:
        st.session_state.prev_lr = learning_rate
    if 'prev_iter' not in st.session_state:
        st.session_state.prev_iter = max_iter
    
    if learning_rate != st.session_state.prev_lr or max_iter != st.session_state.prev_iter:
        st.session_state.full_history = None
        st.session_state.current_step = 0
        st.session_state.prev_lr = learning_rate
        st.session_state.prev_iter = max_iter
        st.info("🔄 Parameters changed! Click any button to run with new settings.")
    
    st.markdown("---")
    st.subheader("🎮 Visualization Controls")
    
    # Button styling with custom CSS
    st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 **Step Forward**", type="primary", use_container_width=True):
            if st.session_state.full_history is None:
                st.session_state.full_history = run_gradient_descent(learning_rate, max_iter)
                st.session_state.current_step = 0
            
            # Increment by step_increment
            new_step = min(st.session_state.current_step + step_increment, len(st.session_state.full_history) - 1)
            st.session_state.current_step = new_step
    
    with col2:
        if st.button("⏮️ **Reset**", use_container_width=True):
            st.session_state.full_history = None
            st.session_state.current_step = 0
    
    if st.button("🎯 **Show Convergence**", type="secondary", use_container_width=True):
        if st.session_state.full_history is None:
            st.session_state.full_history = run_gradient_descent(learning_rate, max_iter)
        st.session_state.current_step = len(st.session_state.full_history) - 1
    
    if st.button("🎬 **Animate Steps**", use_container_width=True):
        if st.session_state.full_history is None:
            st.session_state.full_history = run_gradient_descent(learning_rate, max_iter)
        
        # Animation with larger step increments for performance
        animation_steps = max(1, max_iter // 100)  # Show ~100 frames
        
        # Animation placeholder
        with col_output:
            placeholder = st.empty()
            for step in range(0, len(st.session_state.full_history), animation_steps):
                st.session_state.current_step = step
                current_history = st.session_state.full_history[:step+1]
                fig, loss = create_plot(current_history)
                
                with placeholder.container():
                    st.pyplot(fig)
                    
                    # Use same KPI cards for animation
                    hist_arr = np.array(current_history)
                    prog_pct = (step / max_iter) * 100
                    
                    col_k1, col_k2, col_k3 = st.columns(3)
                    with col_k1:
                        st.markdown(f"""
                        <div class="kpi-card iteration-card">
                            <div class="kpi-title">📊 CURRENT ITERATION</div>
                            <div class="kpi-value">{step:,}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_k2:
                        st.markdown(f"""
                        <div class="kpi-card loss-card">
                            <div class="kpi-title">🎯 MSE LOSS</div>
                            <div class="kpi-value">{loss:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_k3:
                        st.markdown(f"""
                        <div class="kpi-card progress-card">
                            <div class="kpi-title">⚡ PROGRESS</div>
                            <div class="kpi-value">{prog_pct:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    col_k4, col_k5, col_k6 = st.columns(3)
                    with col_k4:
                        st.markdown(f"""
                        <div class="kpi-card param-w-card">
                            <div class="kpi-title">📈 SLOPE (w)</div>
                            <div class="kpi-value">{hist_arr[-1,0]:.5f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_k5:
                        st.markdown(f"""
                        <div class="kpi-card param-b-card">
                            <div class="kpi-title">📉 INTERCEPT (b)</div>
                            <div class="kpi-value">{hist_arr[-1,1]:.5f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_k6:
                        st.markdown(f"""
                        <div class="kpi-card total-card">
                            <div class="kpi-title">🔢 TOTAL ITERATIONS</div>
                            <div class="kpi-value">{max_iter:,}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                plt.close(fig)
                time.sleep(0.03)
            
            # Show final state
            st.session_state.current_step = len(st.session_state.full_history) - 1
            current_history = st.session_state.full_history
            fig, loss = create_plot(current_history)
            
            with placeholder.container():
                st.pyplot(fig)
                
                hist_final = np.array(current_history)
                col_k1, col_k2, col_k3 = st.columns(3)
                with col_k1:
                    st.markdown(f"""
                    <div class="kpi-card iteration-card">
                        <div class="kpi-title">📊 CURRENT ITERATION</div>
                        <div class="kpi-value">{max_iter:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_k2:
                    st.markdown(f"""
                    <div class="kpi-card loss-card">
                        <div class="kpi-title">🎯 MSE LOSS</div>
                        <div class="kpi-value">{loss:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_k3:
                    st.markdown(f"""
                    <div class="kpi-card progress-card">
                        <div class="kpi-title">⚡ PROGRESS</div>
                        <div class="kpi-value">100%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                col_k4, col_k5, col_k6 = st.columns(3)
                with col_k4:
                    st.markdown(f"""
                    <div class="kpi-card param-w-card">
                        <div class="kpi-title">📈 SLOPE (w)</div>
                        <div class="kpi-value">{hist_final[-1,0]:.5f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_k5:
                    st.markdown(f"""
                    <div class="kpi-card param-b-card">
                        <div class="kpi-title">📉 INTERCEPT (b)</div>
                        <div class="kpi-value">{hist_final[-1,1]:.5f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_k6:
                    st.markdown(f"""
                    <div class="kpi-card total-card">
                        <div class="kpi-title">🔢 TOTAL ITERATIONS</div>
                        <div class="kpi-value">{max_iter:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            plt.close(fig)
        
        st.success("✅ Animation complete! Converged to minimum.")

# Display current state
with col_output:
    if st.session_state.full_history is not None:
        current_history = st.session_state.full_history[:st.session_state.current_step+1]
        fig, loss = create_plot(current_history)
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Custom CSS for colorful KPI cards
        st.markdown("""
        <style>
        .kpi-card {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .kpi-title {
            font-size: 14px;
            font-weight: 500;
            opacity: 0.9;
            margin-bottom: 8px;
        }
        .kpi-value {
            font-size: 32px;
            font-weight: 700;
        }
        .iteration-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .loss-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
        .param-w-card { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
        .param-b-card { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
        .progress-card { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
        .total-card { background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); }
        </style>
        """, unsafe_allow_html=True)
        
        hist = np.array(current_history)
        progress_pct = (st.session_state.current_step / max_iter) * 100
        
        # Row 1: Main KPIs
        col_k1, col_k2, col_k3 = st.columns(3)
        
        with col_k1:
            st.markdown(f"""
            <div class="kpi-card iteration-card">
                <div class="kpi-title">📊 CURRENT ITERATION</div>
                <div class="kpi-value">{st.session_state.current_step:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k2:
            st.markdown(f"""
            <div class="kpi-card loss-card">
                <div class="kpi-title">🎯 MSE LOSS</div>
                <div class="kpi-value">{loss:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k3:
            st.markdown(f"""
            <div class="kpi-card progress-card">
                <div class="kpi-title">⚡ PROGRESS</div>
                <div class="kpi-value">{progress_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 2: Parameters
        col_k4, col_k5, col_k6 = st.columns(3)
        
        with col_k4:
            st.markdown(f"""
            <div class="kpi-card param-w-card">
                <div class="kpi-title">📈 SLOPE (w)</div>
                <div class="kpi-value">{hist[-1,0]:.5f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k5:
            st.markdown(f"""
            <div class="kpi-card param-b-card">
                <div class="kpi-title">📉 INTERCEPT (b)</div>
                <div class="kpi-value">{hist[-1,1]:.5f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k6:
            st.markdown(f"""
            <div class="kpi-card total-card">
                <div class="kpi-title">🔢 TOTAL ITERATIONS</div>
                <div class="kpi-value">{max_iter:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar
        progress = st.session_state.current_step / (len(st.session_state.full_history) - 1)
        st.progress(progress)
        
        # Show if converged
        if st.session_state.current_step == len(st.session_state.full_history) - 1:
            st.success("✅ Gradient descent has converged to the minimum!")
    else:
        st.info("👈 Click **Step Forward** to start gradient descent, or **Animate Steps** to watch the full optimization!")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: right; font-size: 0.9em; color: gray;'>"
    "Developed for educational purposes by <strong>Dr. Jishan Ahmed</strong><br>"
    "Department of Mathematics, Weber State University"
    "</div>",
    unsafe_allow_html=True
)
