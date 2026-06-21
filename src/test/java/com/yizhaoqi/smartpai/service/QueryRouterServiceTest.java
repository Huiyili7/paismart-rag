package com.yizhaoqi.smartpai.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yizhaoqi.smartpai.service.QueryRouterService.Route;
import com.yizhaoqi.smartpai.service.QueryRouterService.RoutingResult;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * 路由解析逻辑的单元测试。纯函数，无需网络 / Spring 上下文，可在 CI 直接运行。
 */
class QueryRouterServiceTest {

    private final ObjectMapper mapper = new ObjectMapper();

    /** arguments 为 JSON 字符串（DeepSeek/OpenAI 的常见返回形式），路由到知识库并使用改写后的查询。 */
    @Test
    void parsesKnowledgeBaseRouteWithRewrittenQuery() throws Exception {
        String body = """
            {"choices":[{"message":{"tool_calls":[{"function":{"name":"route_query",
            "arguments":"{\\"destination\\":\\"knowledge_base\\",\\"standalone_query\\":\\"什么是过拟合\\",\\"reason\\":\\"涉及学科知识\\"}"}}]}}]}
            """;
        RoutingResult r = QueryRouterService.parseRouting(mapper, body, "它是什么意思？");
        assertEquals(Route.KNOWLEDGE_BASE, r.route());
        assertEquals("什么是过拟合", r.standaloneQuery());
    }

    /** general_chat 应路由到通用问答。 */
    @Test
    void parsesGeneralChatRoute() throws Exception {
        String body = """
            {"choices":[{"message":{"tool_calls":[{"function":{"name":"route_query",
            "arguments":"{\\"destination\\":\\"general_chat\\",\\"reason\\":\\"寒暄\\"}"}}]}}]}
            """;
        RoutingResult r = QueryRouterService.parseRouting(mapper, body, "你好呀");
        assertEquals(Route.GENERAL_CHAT, r.route());
        // standalone_query 缺省时回退到原始问题
        assertEquals("你好呀", r.standaloneQuery());
    }

    /** arguments 作为内嵌对象（而非字符串）时也能解析。 */
    @Test
    void parsesArgumentsAsNestedObject() throws Exception {
        String body = """
            {"choices":[{"message":{"tool_calls":[{"function":{"name":"route_query",
            "arguments":{"destination":"knowledge_base","standalone_query":"TCP 三次握手"}}}]}}]}
            """;
        RoutingResult r = QueryRouterService.parseRouting(mapper, body, "原问题");
        assertEquals(Route.KNOWLEDGE_BASE, r.route());
        assertEquals("TCP 三次握手", r.standaloneQuery());
    }

    /** 模型未调用函数时，安全回退到知识库检索并保留原始问题。 */
    @Test
    void fallsBackToKnowledgeBaseWhenNoToolCall() throws Exception {
        String body = """
            {"choices":[{"message":{"content":"我直接回答了"}}]}
            """;
        RoutingResult r = QueryRouterService.parseRouting(mapper, body, "二叉树的高度怎么算");
        assertEquals(Route.KNOWLEDGE_BASE, r.route());
        assertEquals("二叉树的高度怎么算", r.standaloneQuery());
    }
}
