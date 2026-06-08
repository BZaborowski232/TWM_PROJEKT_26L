def classify_bottle(
    debris_result,
    fill_result,
    cap_result
):

    # if debris_result["debris"]:

    #     final_class = "debris"
    #     confidence = debris_result["confidence"]
    #     decision_source = "debris"

    # elif fill_result["underfilled"]:

    #     final_class = "underfilled"
    #     confidence = fill_result["confidence"]
    #     decision_source = "fill_level"

    if not cap_result["cap_present"]:

        final_class = "no_cap"
        confidence = cap_result["confidence"]
        decision_source = "cap"

    else:

        final_class = "good"

        confidence = max(
            debris_result["confidence"],
            fill_result["confidence"],
            cap_result["confidence"]
        )

        decision_source = "none"

    return {

        "final_class":
            final_class,

        "confidence":
            round(
                float(confidence),
                3
            ),

        "decision_source":
            decision_source
    }