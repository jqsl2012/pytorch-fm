
import pandas as pd

values = [0, 0.0, 1.0, 0.76056338028169, 0.14103219213081247, 0.0, 0.38156403475632794, 0.12816091954022987, 0.10649350649350649, 0.8082946889491643, 0.125533211456429, 0.11357340720221606, 0.28, 0.05324798713716701, 0.07339955849889625, 0.27588765234265406, 0.0, 0.0, 0.0016626843811010392, 0.007311017783095398, 0.005386399445760076, 0.0074464070019050095, 0.004047986905692898, 0.0008410931304808275, 0.0, 0.0, 0.0, 0.004772551380170264, 0.005480398933543992, 0.005280217482225012, 0.0017335529167027818, 0.005280217482225012, 0.0017335529167027818, 0.0015639662183296842, 0.0016891891891891893, 0.002678511578434348, 0.0041278792099215065, 0.004071203393546055, 0.0041278792099215065, 0.0, 0.0, 0.005434782608695652, 0.0054029502355078035, '35岁以上', '0', '1', '1', '1', '6.0', '613283.0', 2.0, 0.0, 3.0, '34463.0', '130928.0', 719.0, 8929.0, '13448323.0', '20472.0', 4121010.0, '0.0', 10.0, 1.0]
indexs = ['label', 'remarks', 'leaning_lable_value', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'timediff', 'sessions', 'avg_performance_sum', 'avg_count_sum', 'avg_conver_rate', 'avg_roi', 'ct_cost', 'ct_rebates', 'ct_cash_consume', 'ct_no_exposure', 'ct_no_clicks', 'ct_cpc', 'ct_cpc_cost', 'ct_ad_main_account', 'ct_no_leaning', 'ct_no_allocated', 'ct_no_allocated_grown', 'ct_no_allocated_child', 'ct_no_customer_grown', 'ct_no_customer_child', 'ct_no_leaning_repeat_grown', 'ct_no_leaning_repeat_child', 'ct_customer_cost_total', 'ct_customer_cost_grow', 'ct_customer_cost_allocated', 'ct_customer_cost_allocated_grown', 'ct_month_trade_count', 'ct_month_trade_amount', 'ct_trade_count', 'ct_trade_amount', 'reference_age', 'position', 'is_adult', 'gather_type', 'cost_type', 'channel_id', 'creativity_id', 'alloc_person_type', 'is_open_sea', 'put_mode', 'landing_page_link_id', 'area_id', 'subject_id', 'follow_dept_id', 'follower_id', 'course_id', 'promotioner_id', 'receptionist_id', 'enter_way', 'is_sms_verify']

print(len(indexs))
print(len(values))

d = {}
for i in range(len(indexs)):
    d[indexs[i]] = [values[i]]


df = pd.DataFrame(data=d)
print(df.head())