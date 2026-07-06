#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import os
from pycocoevalcap.spice.spice import Spice

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s"
)

def test_spice():
    """SPICE 계산을 테스트하는 함수"""
    
    # 간단한 테스트 데이터
    test_predictions = {
        "test1": ["a man is speaking"],
        "test2": ["music is playing"]
    }
    
    test_references = {
        "test1": ["a person is talking", "someone is speaking"],
        "test2": ["there is music", "musical sound"]
    }
    
    logging.info("SPICE 테스트를 시작합니다...")
    
    try:
        spice_scorer = Spice()
        logging.info("SPICE 스코어러가 성공적으로 초기화되었습니다.")
        
        logging.info("SPICE 계산을 시작합니다...")
        score, scores = spice_scorer.compute_score(test_references, test_predictions)
        
        logging.info(f"SPICE 전체 점수: {score}")
        logging.info(f"개별 점수들: {scores}")
        
        # 점수 형식 확인
        if isinstance(scores, list) and len(scores) > 0:
            logging.info(f"첫 번째 점수 형식: {type(scores[0])}")
            if isinstance(scores[0], dict):
                logging.info(f"딕셔너리 키들: {scores[0].keys()}")
                if "All" in scores[0]:
                    logging.info(f"All 키 내용: {scores[0]['All']}")
        
        return True
        
    except ImportError as e:
        logging.error(f"SPICE 모듈 import 실패: {e}")
        logging.error("pycocoevalcap 패키지가 제대로 설치되지 않았을 수 있습니다.")
        return False
        
    except Exception as e:
        logging.error(f"SPICE 테스트 중 오류 발생: {e}")
        logging.error(f"오류 타입: {type(e)}")
        
        # Java 관련 에러인지 확인
        error_str = str(e).lower()
        if "java" in error_str:
            logging.error("Java 환경 문제일 가능성이 높습니다.")
            logging.error("다음을 확인해주세요:")
            logging.error("1. Java가 설치되어 있는지 확인")
            logging.error("2. JAVA_HOME 환경변수가 설정되어 있는지 확인")
            logging.error("3. 네트워크 연결 상태 확인 (첫 실행 시 파일 다운로드)")
        
        return False

def check_java_environment():
    """Java 환경을 확인하는 함수"""
    logging.info("Java 환경을 확인합니다...")
    
    # Java 버전 확인
    java_version = os.system("java -version")
    if java_version != 0:
        logging.warning("Java가 설치되지 않았거나 PATH에 없습니다.")
        return False
    
    # JAVA_HOME 확인
    java_home = os.environ.get('JAVA_HOME')
    if java_home:
        logging.info(f"JAVA_HOME: {java_home}")
    else:
        logging.warning("JAVA_HOME 환경변수가 설정되지 않았습니다.")
    
    return True

if __name__ == "__main__":
    logging.info("SPICE 디버깅 스크립트를 시작합니다.")
    
    # Java 환경 확인
    check_java_environment()
    
    # SPICE 테스트
    success = test_spice()
    
    if success:
        logging.info("SPICE 테스트가 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        logging.error("SPICE 테스트가 실패했습니다.")
        sys.exit(1) 