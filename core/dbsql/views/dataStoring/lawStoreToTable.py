def full_compare_process(user_file, user_region):
    """完整的合同比较流程"""
    # 1. 解析用户合同
    user_contract = parse_contract(user_file)

    # 2. 检索匹配的标准合同
    template_contracts = search_templates(
        keywords=extract_keywords(user_contract),
        region=user_region
    )

    # 3. 用户选择要比较的模板（此处简化）
    selected_template = load_template(template_contracts[0]['id'])

    # 4. 差异检测
    differences = detect_differences(
        user_contract['clauses'],
        selected_template['clauses']
    )

    # 5. 法律风险分析
    risk_analysis = analyze_legal_risks(differences, user_region)

    # 6. 生成报告
    report = {
        'meta': {
            'user_contract': user_file.name,
            'template_contract': selected_template['title'],
            'compare_date': datetime.now().isoformat()
        },
        'summary': {
            'total_clauses': len(user_contract['clauses']),
            'different_clauses': len(differences),
            'high_risk': sum(1 for r in risk_analysis if r['risk_level'] == '高'),
            'medium_risk': sum(1 for r in risk_analysis if r['risk_level'] == '中')
        },
        'details': risk_analysis
    }

    return report


for user_clause in user_contract:
    best_match = None
    for template_clause in template:
        sim = cosine_similarity(user_clause['embedding'],
                              template_clause['embedding'])
        if sim > 0.7 and (not best_match or sim > best_match['score']):
            best_match = {"id": template_clause['id'], "score": sim}