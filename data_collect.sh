# 自动化收集机器学习训练结果（从 summary.log 文件），按误差类型整理到不同文件中，便于后续分析（如绘图或报告）。
# 脚本依赖 train_record.txt 列出所有训练文件夹路径
function collect() {
	echo $folder >> diag_data.txt
	grep MAE $folder/summary.log -A 6 -B3 | grep "diag " -A 10 >> diag_data.txt
	echo $folder >> close_data.txt
	grep MAE $folder/summary.log -A 6 -B3 | grep "close" -A 10 >> close_data.txt
	echo $folder >> remote_data.txt
	grep MAE $folder/summary.log -A 6 -B3 | grep "remote" -A 10 >> remote_data.txt
	echo $folder >> tot_data.txt
	grep MAE $folder/summary.log -A 6 -B3 | grep "Total" -A 10 >> tot_data.txt
}

for txt_file in diag_data.txt close_data.txt remote_data.txt tot_data.txt;do
	if [[ -f $txt_file ]];then
		rm $txt_file
	fi
done

while IFS= read -r folder; do
    echo "Collect data from: $folder"
	collect
done < train_record.txt

# split output to diag, offdiag_close, and offdiag_remote

