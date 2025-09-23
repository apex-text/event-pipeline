import os
import requests
import logging
from pyspark.sql import DataFrame, functions as F

# 로깅 설정
logger = logging.getLogger(__name__)


def notify_gdelt_anomalies(silver_df: DataFrame):
    """
    GDELT DataFrame에서 avg_tone 이상치를 감지하고 구성된 웹훅(Discord, MS Teams)으로 알림을 보냅니다.
    URL이 중복일 경우 하나로 합쳐서 메시지를 보냅니다.
    """
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    MS_TEAMS_WEBHOOK_URL = os.getenv("MS_TEAMS_WEBHOOK_URL")

    if not DISCORD_WEBHOOK_URL and not MS_TEAMS_WEBHOOK_URL:
        logger.warning("알림을 위한 웹훅 URL이 설정되지 않았습니다. 건너뜁니다.")
        return

    try:
        # avg_tone이 -10 이하인 데이터 필터링
        outliers_df = silver_df.filter(F.col("avg_tone") <= -10).select(
            "global_event_id", "source_url", "avg_tone"
        )

        outliers_count = outliers_df.count()

        if outliers_count > 0:
            logger.info(
                f"📢 {outliers_count}개의 이상치를 발견했습니다. 그룹화하여 알림을 보냅니다..."
            )

            # URL 기준으로 그룹화하고 ID는 리스트로, Tone은 최소값으로 집계
            grouped_outliers_df = outliers_df.groupBy("source_url").agg(
                F.collect_list("global_event_id").alias("event_ids"),
                F.min("avg_tone").alias("min_avg_tone"),
            )

            total_urls = grouped_outliers_df.count()

            # 알림 메시지 생성 (상위 5개 URL만 표시)
            title = f"🚨 GDELT 이벤트 이상치 탐지 ({outliers_count}건 / {total_urls}개 URL) 🚨"
            message_lines = [title]

            outliers_to_show = grouped_outliers_df.limit(5).collect()
            for row in outliers_to_show:
                ids_str = ", ".join(map(str, row["event_ids"]))
                message_lines.append(
                    f"  - IDs: {ids_str}, Tone: {row['min_avg_tone']:.2f}, URL: {row['source_url']}"
                )

            if total_urls > 5:
                message_lines.append(f"  ... 외 {total_urls - 5}개 URL 더 있습니다.")

            message = "\n".join(message_lines)

            # Discord로 알림 보내기
            if DISCORD_WEBHOOK_URL:
                try:
                    # Discord는 <>를 사용하여 URL 임베드를 방지할 수 있습니다.
                    discord_message = message.replace("URL: ", "URL: <") + ">"
                    payload = {"content": discord_message}
                    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
                    response.raise_for_status()
                    logger.info("🚀 Discord 알림을 성공적으로 보냈습니다.")
                except Exception as e:
                    logger.error(
                        f"❌ Discord 알림 전송 중 오류 발생: {e}", exc_info=True
                    )

            # Microsoft Teams로 알림 보내기 (단순 메시지 형식)
            if MS_TEAMS_WEBHOOK_URL:
                try:
                    teams_message = message.replace(
                        "\n", "<br>"
                    )  # Teams는 줄바꿈에 <br> 사용
                    payload = {"content": teams_message}
                    response = requests.post(MS_TEAMS_WEBHOOK_URL, json=payload)
                    response.raise_for_status()
                    logger.info("🚀 Microsoft Teams 알림을 성공적으로 보냈습니다.")
                except Exception as e:
                    logger.error(
                        f"❌ Microsoft Teams 알림 전송 중 오류 발생: {e}", exc_info=True
                    )

        else:
            logger.info("✅ 이상치를 발견하지 않았습니다.")

    except Exception as e:
        logger.error(f"❌ 알림 처리 중 오류 발생: {e}", exc_info=True)
