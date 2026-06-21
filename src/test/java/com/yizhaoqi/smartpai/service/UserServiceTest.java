package com.yizhaoqi.smartpai.service;

import com.yizhaoqi.smartpai.exception.CustomException;
import com.yizhaoqi.smartpai.model.User;
import com.yizhaoqi.smartpai.repository.OrganizationTagRepository;
import com.yizhaoqi.smartpai.repository.UserRepository;
import com.yizhaoqi.smartpai.utils.PasswordUtil;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.http.HttpStatus;

import java.util.Optional;

import static org.hamcrest.Matchers.any;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;
/**
 * UserService 的测试类
 */
class UserServiceTest {
    // 模拟 UserRepository 实例
    @Mock
    private UserRepository userRepository;

    // UserService 还依赖组织标签仓库与标签缓存服务，需一并 mock，否则注入为 null
    @Mock
    private OrganizationTagRepository organizationTagRepository;

    @Mock
    private OrgTagCacheService orgTagCacheService;

    // 注入模拟的 UserService 实例
    @InjectMocks
    private UserService userService;

    /**
     * 在每个测试方法执行前初始化模拟对象
     */
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    /**
     * 测试用户注册成功的情况
     */
    @Test
    void testRegisterUser_Success() {
        // 假设用户名 "testuser" 在数据库中不存在
        when(userRepository.findByUsername("testuser")).thenReturn(Optional.empty());
        // 组织标签已存在，跳过默认/私人标签的创建分支，聚焦于用户注册本身
        when(organizationTagRepository.existsByTagId(anyString())).thenReturn(true);

        // 调用 userService 的 registerUser 方法进行用户注册
        userService.registerUser("testuser", "password123");

        // 创建 ArgumentCaptor 来捕获 save 方法的参数
        ArgumentCaptor<User> userCaptor = ArgumentCaptor.forClass(User.class);

        // registerUser 会先保存用户以生成 ID、再保存组织标签信息，故 save 会被调用多次；
        // 这里验证至少保存过，并捕获最终落库的用户对象
        verify(userRepository, atLeastOnce()).save(userCaptor.capture());

        // 获取捕获的 User 对象并进行断言
        User savedUser = userCaptor.getValue();
        assertNotNull(savedUser);
        assertEquals("testuser", savedUser.getUsername());
    }

    /**
     * 测试用户注册时用户名已存在的情况
     */
    @Test
    void testRegisterUser_UsernameExists() {
        // 假设用户名 "testuser" 在数据库中已存在
        when(userRepository.findByUsername("testuser")).thenReturn(Optional.of(new User()));

        // 断言在注册已存在的用户名时抛出 CustomException 异常
        CustomException exception = assertThrows(CustomException.class, () -> userService.registerUser("testuser", "password123"));
        assertEquals("Username already exists", exception.getMessage());
        assertEquals(HttpStatus.BAD_REQUEST, exception.getStatus());
    }

    /**
     * 测试用户认证成功的情况
     */
   @Test
void testAuthenticateUser_Success() {
    // 使用 PasswordUtil 生成加密密码
    String rawPassword = "password123";
    String encodedPassword = PasswordUtil.encode(rawPassword);

    // 创建一个带有加密密码的用户对象，并确保设置了用户名
    User user = new User();
    user.setUsername("testuser"); // 确保设置了用户名
    user.setPassword(encodedPassword);

    when(userRepository.findByUsername("testuser")).thenReturn(Optional.of(user));

    try {
        // 调用 userService 的 authenticateUser 方法进行用户认证
        String username = userService.authenticateUser("testuser", rawPassword);

        // 打印返回的用户名（仅用于调试）
        System.out.println("Returned username: " + username);

        // 断言返回的用户名是否正确
        assertEquals("testuser", username);
    } catch (CustomException e) {
        // 捕获并打印异常信息
        System.out.println("Exception message: " + e.getMessage());
        throw e;
    }

    // 打印实际的加密密码（仅用于调试）
    System.out.println("Actual encrypted password: " + user.getPassword());
}





    /**
     * 测试用户认证失败的情况
     */
    @Test
    void testAuthenticateUser_InvalidCredentials() {
        // 假设用户名 "testuser" 在数据库中不存在
        when(userRepository.findByUsername("testuser")).thenReturn(Optional.empty());

        // 断言在使用错误密码认证时抛出 CustomException 异常
        CustomException exception = assertThrows(CustomException.class, () -> userService.authenticateUser("testuser", "wrongpassword"));
        assertEquals("Invalid username or password", exception.getMessage());
        assertEquals(HttpStatus.UNAUTHORIZED, exception.getStatus());
    }
}
