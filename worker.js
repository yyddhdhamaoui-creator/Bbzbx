export default {
  async fetch(request, env) {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    if (request.method !== "POST") {
      return new Response(JSON.stringify({ error: "Method not allowed" }), {
        status: 405,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    try {
      const url = new URL(request.url);
      const modelPath = url.pathname.replace(/^\/+|\/+$/g, "");

      if (!modelPath) {
        return new Response(
          JSON.stringify({
            error: "No model specified",
            example: "/imager-1-pro",
          }),
          { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      const repoId = `Imagers/${modelPath}`;

      const formData = await request.formData();
      const file = formData.get("file");

      if (!file) {
        return new Response(JSON.stringify({ error: "No file provided" }), {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      const arrayBuffer = await file.arrayBuffer();

      const base = env.HF_BASE_URL || "https://router.huggingface.co/hf-inference/models";
      const hfUrl = `${base}/${repoId}`;

      const hfResponse = await fetch(hfUrl, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.HF_TOKEN}`,
          "Content-Type": "application/octet-stream",
        },
        body: arrayBuffer,
      });

      if (!hfResponse.ok) {
        const errorText = await hfResponse.text();
        let friendlyError = "Model error";
        if (hfResponse.status === 503) friendlyError = "Model is loading, retry in 20 seconds";
        if (hfResponse.status === 404) friendlyError = `Model '${repoId}' not found`;
        if (hfResponse.status === 401) friendlyError = "Invalid HF token";

        return new Response(
          JSON.stringify({ error: friendlyError, model: repoId, status: hfResponse.status, detail: errorText.slice(0, 300) }),
          { status: hfResponse.status, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      const result = await hfResponse.json();
      let aiScore = 0, realScore = 0;
      for (const item of result) {
        const label = (item.label || "").toLowerCase();
        if (label.includes("ai") || label.includes("fake") || label.includes("generated")) aiScore = item.score;
        else if (label.includes("real")) realScore = item.score;
      }

      const isAI = aiScore > realScore;
      const confidence = (isAI ? aiScore : realScore) * 100;

      return new Response(
        JSON.stringify({
          model_repo: repoId,
          label: isAI ? "AI-Generated" : "Real Image",
          isAI: isAI,
          confidence: parseFloat(confidence.toFixed(2)),
          aiScore: parseFloat((aiScore * 100).toFixed(2)),
          realScore: parseFloat((realScore * 100).toFixed(2)),
        }),
        { status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    } catch (error) {
      return new Response(
        JSON.stringify({ error: "Internal error", detail: error.message }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
  },
};
