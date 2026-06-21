package com.yizhaoqi.smartpai.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.List;
import java.util.Map;

/**
 * 基于大模型 Function Calling 的查询路由服务。
 *
 * <p>在执行 RAG 检索之前，先用一次 function-calling 调用让模型判断：当前用户问题
 * 是否需要查询知识库（{@link Route#KNOWLEDGE_BASE}），还是属于寒暄 / 通用问答，
 * 不必检索（{@link Route#GENERAL_CHAT}）。同时让模型把可能依赖上下文的问题改写成
 * 一个独立、可检索的查询（standalone query），提升多轮场景下的召回质量。</p>
 *
 * <p>设计上对失败保持「安全默认」：任何异常都回退到 {@link Route#KNOWLEDGE_BASE}，
 * 保证宁可多检索也不漏检索，绝不因路由层故障而阻断主链路。</p>
 */
@Service
public class QueryRouterService {

    private static final Logger logger = LoggerFactory.getLogger(QueryRouterService.class);

    /** 路由目标：是否需要查询知识库。 */
    public enum Route {
        KNOWLEDGE_BASE,
        GENERAL_CHAT
    }

    /**
     * 路由结果。
     *
     * @param route          目标路由
     * @param standaloneQuery 用于检索的独立查询（可能是对原问题的改写）
     * @param reason         模型给出的判断理由（便于排查 / 审计）
     */
    public record RoutingResult(Route route, String standaloneQuery, String reason) {
    }

    private final WebClient webClient;
    private final String model;
    private final ObjectMapper objectMapper;

    public QueryRouterService(@Value("${deepseek.api.url}") String apiUrl,
                              @Value("${deepseek.api.key}") String apiKey,
                              @Value("${deepseek.api.model}") String model) {
        WebClient.Builder builder = WebClient.builder().baseUrl(apiUrl);
        if (apiKey != null && !apiKey.trim().isEmpty()) {
            builder.defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey);
        }
        this.webClient = builder.build();
        this.model = model;
        this.objectMapper = new ObjectMapper();
    }

    /**
     * 对用户问题做路由。失败时安全回退到知识库检索。
     */
    public RoutingResult route(String userMessage) {
        try {
            Map<String, Object> request = buildRequest(userMessage);
            String response = webClient.post()
                    .uri("/chat/completions")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(request)
                    .retrieve()
                    .bodyToMono(String.class)
                    .block(Duration.ofSeconds(15));
            RoutingResult result = parseRouting(objectMapper, response, userMessage);
            logger.info("查询路由结果: route={}, standaloneQuery={}, reason={}",
                    result.route(), result.standaloneQuery(), result.reason());
            return result;
        } catch (Exception e) {
            logger.warn("查询路由失败，回退到知识库检索: {}", e.getMessage());
            return fallback(userMessage);
        }
    }

    private Map<String, Object> buildRequest(String userMessage) {
        // 路由函数：让模型把决策以结构化参数返回
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of(
                        "destination", Map.of(
                                "type", "string",
                                "enum", List.of("knowledge_base", "general_chat"),
                                "description", "knowledge_base=需要查询教育知识库; general_chat=寒暄或通用问答，无需检索"),
                        "standalone_query", Map.of(
                                "type", "string",
                                "description", "改写后的、可独立检索的查询；若无需检索可留空"),
                        "reason", Map.of(
                                "type", "string",
                                "description", "做出该路由判断的简短理由")),
                "required", List.of("destination"));

        Map<String, Object> function = Map.of(
                "name", "route_query",
                "description", "根据用户问题判断是否需要检索知识库，并给出用于检索的独立查询",
                "parameters", parameters);

        Map<String, Object> tool = Map.of("type", "function", "function", function);

        String systemPrompt = "你是查询路由器。判断用户问题是否需要查询教育知识库。"
                + "涉及课程、学科知识、文档内容的问题选择 knowledge_base；"
                + "纯寒暄、问候、与知识库无关的闲聊选择 general_chat。"
                + "请始终通过调用 route_query 函数返回结构化结果。";

        return Map.of(
                "model", model,
                "messages", List.of(
                        Map.of("role", "system", "content", systemPrompt),
                        Map.of("role", "user", "content", userMessage)),
                "tools", List.of(tool),
                // 强制模型调用 route_query，保证一定返回结构化结果
                "tool_choice", Map.of("type", "function", "function", Map.of("name", "route_query")),
                "temperature", 0,
                "stream", false);
    }

    /**
     * 解析 function-calling 响应（纯函数，便于单测）。
     *
     * @param mapper       JSON 解析器
     * @param responseBody chat/completions 的原始响应体
     * @param originalQuery 原始用户问题，作为 standaloneQuery 的兜底
     */
    static RoutingResult parseRouting(ObjectMapper mapper, String responseBody, String originalQuery)
            throws Exception {
        JsonNode root = mapper.readTree(responseBody);
        JsonNode toolCalls = root.path("choices").path(0).path("message").path("tool_calls");
        if (!toolCalls.isArray() || toolCalls.isEmpty()) {
            // 模型没有按预期调用函数，安全回退
            return new RoutingResult(Route.KNOWLEDGE_BASE, originalQuery, "no tool_call, default to KB");
        }

        JsonNode arguments = toolCalls.path(0).path("function").path("arguments");
        // arguments 通常是一个 JSON 字符串，需要二次解析
        JsonNode args = arguments.isTextual() ? mapper.readTree(arguments.asText()) : arguments;

        String destination = args.path("destination").asText("knowledge_base");
        String standalone = args.path("standalone_query").asText("");
        String reason = args.path("reason").asText("");

        Route route = "general_chat".equalsIgnoreCase(destination) ? Route.GENERAL_CHAT : Route.KNOWLEDGE_BASE;
        String query = (standalone == null || standalone.isBlank()) ? originalQuery : standalone;
        return new RoutingResult(route, query, reason);
    }

    private RoutingResult fallback(String userMessage) {
        return new RoutingResult(Route.KNOWLEDGE_BASE, userMessage, "router error fallback");
    }
}
