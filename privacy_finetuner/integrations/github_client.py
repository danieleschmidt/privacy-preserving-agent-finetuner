"""GitHub API client for repository management and automation."""

import asyncio
import logging
from typing import Optional, Dict, List, Any
import aiohttp
import base64
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class GitHubClient:
    """GitHub API client for repository operations."""
    
    def __init__(self, token: str, base_url: str = "https://api.github.com"):
        """Initialize GitHub client.
        
        Args:
            token: GitHub personal access token or app token
            base_url: GitHub API base URL (for enterprise)
        """
        self.token = token
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Privacy-Finetuner/1.0.0"
        }
        
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            connector=connector,
            timeout=timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to GitHub API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            API response data
            
        Raises:
            aiohttp.ClientError: Request failed
        """
        if not self.session:
            raise RuntimeError("GitHub client not initialized. Use async context manager.")
            
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                
                if response.content_type == "application/json":
                    return await response.json()
                else:
                    return {"content": await response.text()}
                    
        except aiohttp.ClientError as e:
            logger.error(f"GitHub API request failed: {method} {url} - {e}")
            raise
            
    async def get_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository data
        """
        return await self._request("GET", f"repos/{owner}/{repo}")
        
    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a new issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Issue title
            body: Issue body
            labels: Issue labels
            assignees: Issue assignees
            
        Returns:
            Created issue data
        """
        data = {
            "title": title,
            "body": body
        }
        
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
            
        return await self._request("POST", f"repos/{owner}/{repo}/issues", json=data)
        
    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: str,
        draft: bool = False
    ) -> Dict[str, Any]:
        """Create a pull request.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: PR title
            head: Head branch
            base: Base branch
            body: PR body
            draft: Whether to create as draft
            
        Returns:
            Created pull request data
        """
        data = {
            "title": title,
            "head": head,
            "base": base,
            "body": body,
            "draft": draft
        }
        
        return await self._request("POST", f"repos/{owner}/{repo}/pulls", json=data)
        
    async def get_workflow_runs(
        self,
        owner: str,
        repo: str,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        per_page: int = 30
    ) -> Dict[str, Any]:
        """Get workflow runs.
        
        Args:
            owner: Repository owner
            repo: Repository name
            workflow_id: Specific workflow ID
            status: Filter by status (queued, in_progress, completed)
            per_page: Results per page
            
        Returns:
            Workflow runs data
        """
        endpoint = f"repos/{owner}/{repo}/actions/runs"
        params = {"per_page": per_page}
        
        if workflow_id:
            endpoint = f"repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs"
        if status:
            params["status"] = status
            
        return await self._request("GET", endpoint, params=params)
        
    async def create_deployment(
        self,
        owner: str,
        repo: str,
        ref: str,
        environment: str,
        description: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a deployment.
        
        Args:
            owner: Repository owner
            repo: Repository name
            ref: Git ref to deploy
            environment: Target environment
            description: Deployment description
            payload: Deployment payload
            
        Returns:
            Created deployment data
        """
        data = {
            "ref": ref,
            "environment": environment,
            "auto_merge": False,
            "required_contexts": []
        }
        
        if description:
            data["description"] = description
        if payload:
            data["payload"] = payload
            
        return await self._request("POST", f"repos/{owner}/{repo}/deployments", json=data)
        
    async def create_deployment_status(
        self,
        owner: str,
        repo: str,
        deployment_id: int,
        state: str,
        target_url: Optional[str] = None,
        description: Optional[str] = None,
        environment_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update deployment status.
        
        Args:
            owner: Repository owner
            repo: Repository name
            deployment_id: Deployment ID
            state: Status (pending, success, failure, error, inactive)
            target_url: Target URL
            description: Status description
            environment_url: Environment URL
            
        Returns:
            Deployment status data
        """
        data = {"state": state}
        
        if target_url:
            data["target_url"] = target_url
        if description:
            data["description"] = description
        if environment_url:
            data["environment_url"] = environment_url
            
        endpoint = f"repos/{owner}/{repo}/deployments/{deployment_id}/statuses"
        return await self._request("POST", endpoint, json=data)
        
    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get file content from repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            ref: Git reference (branch, tag, commit)
            
        Returns:
            File content data
        """
        endpoint = f"repos/{owner}/{repo}/contents/{path}"
        params = {}
        
        if ref:
            params["ref"] = ref
            
        response = await self._request("GET", endpoint, params=params)
        
        # Decode base64 content if present
        if "content" in response and response.get("encoding") == "base64":
            try:
                response["decoded_content"] = base64.b64decode(
                    response["content"].replace('\n', '')
                ).decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to decode file content: {e}")
                
        return response
        
    async def create_or_update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: Optional[str] = None,
        sha: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create or update a file in repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            content: File content
            message: Commit message
            branch: Target branch
            sha: Current file SHA (for updates)
            
        Returns:
            File operation response
        """
        # Encode content to base64
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        data = {
            "message": message,
            "content": encoded_content
        }
        
        if branch:
            data["branch"] = branch
        if sha:
            data["sha"] = sha
            
        endpoint = f"repos/{owner}/{repo}/contents/{path}"
        return await self._request("PUT", endpoint, json=data)
        
    async def get_releases(
        self,
        owner: str,
        repo: str,
        per_page: int = 30
    ) -> List[Dict[str, Any]]:
        """Get repository releases.
        
        Args:
            owner: Repository owner
            repo: Repository name
            per_page: Results per page
            
        Returns:
            List of releases
        """
        params = {"per_page": per_page}
        response = await self._request("GET", f"repos/{owner}/{repo}/releases", params=params)
        return response if isinstance(response, list) else []
        
    async def create_release(
        self,
        owner: str,
        repo: str,
        tag_name: str,
        name: Optional[str] = None,
        body: Optional[str] = None,
        target_commitish: Optional[str] = None,
        draft: bool = False,
        prerelease: bool = False
    ) -> Dict[str, Any]:
        """Create a new release.
        
        Args:
            owner: Repository owner
            repo: Repository name
            tag_name: Tag name
            name: Release name
            body: Release description
            target_commitish: Target commit/branch
            draft: Create as draft
            prerelease: Mark as prerelease
            
        Returns:
            Created release data
        """
        data = {
            "tag_name": tag_name,
            "draft": draft,
            "prerelease": prerelease
        }
        
        if name:
            data["name"] = name
        if body:
            data["body"] = body
        if target_commitish:
            data["target_commitish"] = target_commitish
            
        return await self._request("POST", f"repos/{owner}/{repo}/releases", json=data)
        
    async def get_repository_events(
        self,
        owner: str,
        repo: str,
        per_page: int = 30
    ) -> List[Dict[str, Any]]:
        """Get repository events.
        
        Args:
            owner: Repository owner
            repo: Repository name
            per_page: Results per page
            
        Returns:
            List of repository events
        """
        params = {"per_page": per_page}
        response = await self._request("GET", f"repos/{owner}/{repo}/events", params=params)
        return response if isinstance(response, list) else []
        
    async def search_code(
        self,
        query: str,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        per_page: int = 30
    ) -> Dict[str, Any]:
        """Search code across repositories.
        
        Args:
            query: Search query
            sort: Sort field (indexed, updated)
            order: Sort order (asc, desc)
            per_page: Results per page
            
        Returns:
            Search results
        """
        params = {
            "q": query,
            "per_page": per_page
        }
        
        if sort:
            params["sort"] = sort
        if order:
            params["order"] = order
            
        return await self._request("GET", "search/code", params=params)
        
    async def get_rate_limit(self) -> Dict[str, Any]:
        """Get API rate limit status.
        
        Returns:
            Rate limit information
        """
        return await self._request("GET", "rate_limit")


class GitHubWebhookHandler:
    """Handle GitHub webhook events."""
    
    def __init__(self, secret: Optional[str] = None):
        """Initialize webhook handler.
        
        Args:
            secret: Webhook secret for signature verification
        """
        self.secret = secret
        
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature.
        
        Args:
            payload: Request payload
            signature: GitHub signature header
            
        Returns:
            True if signature is valid
        """
        if not self.secret:
            logger.warning("No webhook secret configured")
            return True
            
        import hmac
        import hashlib
        
        expected_signature = hmac.new(
            self.secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # GitHub prefixes signature with 'sha256='
        signature = signature.replace('sha256=', '') if signature.startswith('sha256=') else signature
        
        return hmac.compare_digest(expected_signature, signature)
        
    async def handle_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle webhook event.
        
        Args:
            event_type: GitHub event type
            payload: Event payload
            
        Returns:
            Event handling result
        """
        handler_method = f"handle_{event_type}"
        
        if hasattr(self, handler_method):
            handler = getattr(self, handler_method)
            return await handler(payload)
        else:
            logger.info(f"No handler for event type: {event_type}")
            return {"status": "ignored", "event_type": event_type}
            
    async def handle_push(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle push event."""
        ref = payload.get("ref", "")
        commits = payload.get("commits", [])
        
        logger.info(f"Push event: {ref} with {len(commits)} commits")
        
        return {
            "status": "processed",
            "event_type": "push",
            "ref": ref,
            "commit_count": len(commits)
        }
        
    async def handle_pull_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pull request event."""
        action = payload.get("action")
        pr = payload.get("pull_request", {})
        pr_number = pr.get("number")
        
        logger.info(f"Pull request event: {action} for PR #{pr_number}")
        
        return {
            "status": "processed",
            "event_type": "pull_request",
            "action": action,
            "pr_number": pr_number
        }
        
    async def handle_workflow_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow run event."""
        action = payload.get("action")
        workflow_run = payload.get("workflow_run", {})
        conclusion = workflow_run.get("conclusion")
        
        logger.info(f"Workflow run event: {action} with conclusion: {conclusion}")
        
        return {
            "status": "processed",
            "event_type": "workflow_run",
            "action": action,
            "conclusion": conclusion
        }