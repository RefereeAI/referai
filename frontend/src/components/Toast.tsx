import { Check, OctagonAlert } from "lucide-react";
import React from "react";

interface ToastProps {
  message: string;
  type: "error" | "success";
  onClose: () => void;
}

const Toast: React.FC<ToastProps> = ({ message, type, onClose }) => {
  const bgColor = type === "error" ? "bg-red-500" : "bg-green-500";
  const icon = type === "error" ? <OctagonAlert className="w-5 h-5"/> : <Check className="w-5 h-5"/>;

  return (
    <div
      className={`fixed top-4 right-4 w-80 p-4 rounded-lg shadow-lg text-white ${bgColor} flex items-center space-x-3 z-50`}
    >
      <span className="text-xl">{icon}</span>
      <span className="flex-1">{message}</span>
      <button
        onClick={onClose}
        className="text-lg font-bold hover:bg-opacity-80"
      >
        ✕
      </button>
    </div>
  );
};

export default Toast;
